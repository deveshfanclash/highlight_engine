"""
Frame Distributor

Thread-based component that reads frames and distributes them to worker queues.
Implements batch-level round-robin distribution for optimal GPU cache utilization.

Distribution Formula:
    batch_id = frame_number // batch_size
    worker_id = batch_id % num_workers

This ensures all frames from the same batch go to the same worker.

Usage:
    distributor = FrameDistributor(
        frame_source=input_handler.iterate(),
        worker_pool=pool,
        batch_size=8,
    )
    distributor.start()

    # ... workers process frames ...

    distributor.wait()  # Wait for all frames to be distributed
"""

import logging
from threading import Thread, Event
from typing import Iterator, List, Dict, Any, Optional
from dataclasses import dataclass
import time

from core.worker_pool import WorkerPool, WorkItem

logger = logging.getLogger(__name__)


@dataclass
class DistributorStats:
    """Statistics from frame distribution."""
    frames_distributed: int = 0
    batches_distributed: int = 0
    elapsed_seconds: float = 0.0
    frames_per_second: float = 0.0


class FrameDistributor:
    """
    Distributes frames from a source to worker queues.

    Features:
    - Batch-level round-robin distribution
    - Auto-detects stream type for buffering
    - Sends None sentinels on completion for worker shutdown
    - Thread-based for non-blocking operation

    Distribution Strategy:
        Frames are grouped into batches, and each batch goes to one worker.
        batch_id = frame_number // batch_size
        worker_id = batch_id % num_workers

        This ensures:
        1. All frames in a batch go to the same worker (GPU cache friendly)
        2. Work is evenly distributed across workers
        3. Deterministic mapping (same frame always goes to same worker)
    """

    def __init__(
        self,
        frame_source: Iterator,
        worker_pool: WorkerPool,
        batch_size: int = 1,
    ):
        """
        Initialize frame distributor.

        Args:
            frame_source: Iterator yielding FrameInputPacket objects
            worker_pool: WorkerPool to distribute to
            batch_size: Frames per batch (for distribution formula)
        """
        self.frame_source = frame_source
        self.worker_pool = worker_pool
        self.batch_size = batch_size
        self.num_workers = worker_pool.num_workers

        self._thread: Optional[Thread] = None
        self._running = False
        self._stopped = Event()

        # Statistics
        self._frames_distributed = 0
        self._batches_distributed = 0
        self._start_time: Optional[float] = None

        # Current batch accumulator
        self._current_batch_frames: List[Any] = []
        self._current_batch_metadata: List[Dict[str, Any]] = []
        self._current_batch_id = 0

    def _distribute_loop(self):
        """Main distribution loop (runs in thread)."""
        self._start_time = time.perf_counter()
        frame_number = 0

        try:
            for frame_packet in self.frame_source:
                if not self._running:
                    break

                # Extract frame and metadata
                frame = frame_packet.frame
                metadata = {
                    "sequence_number": frame_packet.sequence_number,
                    "timestamp_ms": frame_packet.timestamp_ms,
                    "width": frame_packet.width,
                    "height": frame_packet.height,
                    "segment_number": getattr(frame_packet, "segment_number", 0),
                }

                # Add to current batch
                self._current_batch_frames.append(frame)
                self._current_batch_metadata.append(metadata)
                frame_number += 1

                # Check if batch is complete
                if len(self._current_batch_frames) >= self.batch_size:
                    self._submit_current_batch()

            # Submit any remaining frames in partial batch
            if self._current_batch_frames:
                self._submit_current_batch()

            self._frames_distributed = frame_number
            logger.info(f"FrameDistributor finished: {frame_number} frames distributed")

        except Exception as e:
            logger.error(f"Error in frame distribution: {e}")

        finally:
            # Send None sentinels to all workers to signal completion
            self._send_shutdown_sentinels()
            self._stopped.set()

    def _submit_current_batch(self):
        """Submit the current accumulated batch to appropriate worker."""
        if not self._current_batch_frames:
            return

        # Calculate worker based on batch ID
        worker_id = self._current_batch_id % self.num_workers

        # Create work item
        work_item = WorkItem(
            batch_id=self._current_batch_id,
            frames=self._current_batch_frames,
            metadata=self._current_batch_metadata,
        )

        # Submit to worker
        while self._running:
            if self.worker_pool.submit(worker_id, work_item):
                break
            # Queue full, wait and retry
            time.sleep(0.01)

        self._batches_distributed += 1
        self._current_batch_id += 1

        # Clear for next batch
        self._current_batch_frames = []
        self._current_batch_metadata = []

    def _send_shutdown_sentinels(self):
        """Send None to each worker queue to signal shutdown."""
        logger.info("Sending shutdown sentinels to workers")
        for i in range(self.num_workers):
            try:
                self.worker_pool._input_queues[i].put(None, timeout=5.0)
            except Exception as e:
                logger.warning(f"Could not send sentinel to worker {i}: {e}")

    def start(self):
        """Start frame distribution in background thread."""
        if self._running:
            logger.warning("FrameDistributor already running")
            return

        self._running = True
        self._stopped.clear()
        self._thread = Thread(target=self._distribute_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"FrameDistributor started: batch_size={self.batch_size}, "
            f"num_workers={self.num_workers}"
        )

    def stop(self):
        """Signal distributor to stop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._stopped.wait(timeout=5.0)
            self._thread.join(timeout=2.0)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for distribution to complete.

        Args:
            timeout: Max time to wait (None = forever)

        Returns:
            True if completed, False if timeout
        """
        return self._stopped.wait(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return self._running and not self._stopped.is_set()

    @property
    def is_done(self) -> bool:
        return self._stopped.is_set()

    @property
    def stats(self) -> DistributorStats:
        """Get distribution statistics."""
        elapsed = 0.0
        fps = 0.0
        if self._start_time:
            elapsed = time.perf_counter() - self._start_time
            fps = self._frames_distributed / elapsed if elapsed > 0 else 0

        return DistributorStats(
            frames_distributed=self._frames_distributed,
            batches_distributed=self._batches_distributed,
            elapsed_seconds=elapsed,
            frames_per_second=fps,
        )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from dataclasses import dataclass
    import numpy as np

    # Mock FrameInputPacket
    @dataclass
    class MockFramePacket:
        frame: np.ndarray
        sequence_number: int
        timestamp_ms: int
        width: int = 100
        height: int = 100
        segment_number: int = 0

    # Mock frame source
    def mock_frame_source(num_frames: int):
        for i in range(num_frames):
            yield MockFramePacket(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                sequence_number=i,
                timestamp_ms=i * 40,
            )

    # Mock WorkerPool
    class MockWorkerPool:
        def __init__(self, num_workers: int):
            self.num_workers = num_workers
            self._input_queues = [[] for _ in range(num_workers)]
            self._submitted = []

        def submit(self, worker_id: int, work_item: WorkItem) -> bool:
            self._submitted.append((worker_id, work_item))
            self._input_queues[worker_id].append(work_item)
            return True

    print("Testing FrameDistributor...")

    # Test basic distribution
    pool = MockWorkerPool(num_workers=4)
    source = mock_frame_source(num_frames=32)

    distributor = FrameDistributor(
        frame_source=source,
        worker_pool=pool,
        batch_size=4,
    )

    distributor.start()
    distributor.wait(timeout=5.0)

    print(f"  Frames distributed: {distributor.stats.frames_distributed}")
    print(f"  Batches distributed: {distributor.stats.batches_distributed}")

    # Verify distribution
    for worker_id in range(4):
        batches = pool._input_queues[worker_id]
        batch_ids = [b.batch_id for b in batches if b is not None]
        print(f"  Worker {worker_id} got batches: {batch_ids}")

    # Verify batch distribution formula
    expected = {
        0: [0, 4],  # batch_id % 4 == 0
        1: [1, 5],  # batch_id % 4 == 1
        2: [2, 6],  # batch_id % 4 == 2
        3: [3, 7],  # batch_id % 4 == 3
    }

    for worker_id, expected_batches in expected.items():
        actual = [b.batch_id for b in pool._input_queues[worker_id] if b is not None]
        assert actual == expected_batches, f"Worker {worker_id}: expected {expected_batches}, got {actual}"

    print("\nAll tests passed!")
