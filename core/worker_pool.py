"""
Worker Pool

Process-based worker pool for parallel inference.
Each worker runs in a separate process for true GPU parallelism.

Architecture:
    Main Process                    Worker Processes
    ============                    ================
    FrameDistributor  ------>       Worker 0 (GPU 0)  ------>
                      ------>       Worker 1 (GPU 1)  ------> Output Queue
                      ------>       Worker 2 (GPU 0)  ------>
                      ------>       Worker 3 (GPU 1)  ------>

Usage:
    pool = WorkerPool(
        num_workers=4,
        worker_fn=my_worker_function,
        worker_init_fn=my_init_function,
        worker_init_args={'model_path': '/path/to/model'},
    )
    pool.start()

    # Send work to workers
    for batch in batches:
        pool.submit(worker_id, batch)

    # Get results
    for result in pool.results():
        process_result(result)

    pool.stop()
"""

import logging
import multiprocessing as mp
from multiprocessing import Process, Queue
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, List, Iterator
from queue import Empty
import signal
import time

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for a single worker."""
    worker_id: int
    device: str  # e.g., "cuda:0", "cuda:1"
    model_path: str
    model_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkItem:
    """Item sent to worker for processing."""
    batch_id: int
    frames: List[Any]  # List of numpy arrays
    metadata: List[Dict[str, Any]]  # Metadata for each frame


@dataclass
class WorkResult:
    """Result from worker processing."""
    batch_id: int
    worker_id: int
    results: List[Optional[Dict[str, Any]]]  # Result for each frame
    processing_time_ms: float
    error: Optional[str] = None


class InferenceWorker:
    """
    Worker that runs in a child process.

    Loads model once at startup, then processes batches from input queue.
    """

    @staticmethod
    def run(
        worker_id: int,
        input_queue: Queue,
        output_queue: Queue,
        worker_init_fn: Callable,
        worker_process_fn: Callable,
        init_args: Dict[str, Any],
    ):
        """
        Worker process main function.

        Args:
            worker_id: Unique identifier for this worker
            input_queue: Queue to receive work items from
            output_queue: Queue to send results to
            worker_init_fn: Function to initialize model (called once)
            worker_process_fn: Function to process batches
            init_args: Arguments for worker_init_fn
        """
        # Ignore SIGINT in worker - let main process handle it
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        logger.info(f"Worker {worker_id} starting, device={init_args.get('device', 'cpu')}")

        model = None
        try:
            # Initialize model
            model = worker_init_fn(**init_args)
            logger.info(f"Worker {worker_id} model loaded")

            # Process work items until sentinel
            while True:
                try:
                    work_item = input_queue.get(timeout=1.0)

                    if work_item is None:
                        # Sentinel - time to exit
                        logger.info(f"Worker {worker_id} received shutdown signal")
                        break

                    # Process the batch
                    start_time = time.perf_counter()
                    try:
                        results = worker_process_fn(
                            model=model,
                            frames=work_item.frames,
                            metadata=work_item.metadata,
                        )
                        processing_time = (time.perf_counter() - start_time) * 1000

                        output_queue.put(WorkResult(
                            batch_id=work_item.batch_id,
                            worker_id=worker_id,
                            results=results,
                            processing_time_ms=processing_time,
                        ))

                    except Exception as e:
                        logger.error(f"Worker {worker_id} error processing batch {work_item.batch_id}: {e}")
                        output_queue.put(WorkResult(
                            batch_id=work_item.batch_id,
                            worker_id=worker_id,
                            results=[None] * len(work_item.frames),
                            processing_time_ms=0,
                            error=str(e),
                        ))

                except Empty:
                    continue

        except Exception as e:
            logger.error(f"Worker {worker_id} fatal error: {e}")

        finally:
            logger.info(f"Worker {worker_id} exiting")
            # Cleanup model if needed
            del model


class WorkerPool:
    """
    Manages pool of inference worker processes.

    Features:
    - Process-based workers for true GPU parallelism
    - Batch-level frame distribution
    - Graceful shutdown with timeout + force terminate
    - Result collection from output queue

    Usage:
        pool = WorkerPool(
            num_workers=4,
            worker_init_fn=ODService.worker_init,
            worker_process_fn=ODService.worker_process_batch,
            worker_configs=[
                {'device': 'cuda:0', 'model_path': '/path/to/model'},
                {'device': 'cuda:1', 'model_path': '/path/to/model'},
                ...
            ],
            queue_size=16,
        )
        pool.start()

        # Submit work
        pool.submit(0, work_item)  # To worker 0

        # Get results
        for result in pool.results():
            handle_result(result)

        pool.stop()
    """

    def __init__(
        self,
        num_workers: int,
        worker_init_fn: Callable,
        worker_process_fn: Callable,
        worker_configs: List[Dict[str, Any]],
        queue_size: int = 0,  # 0 = auto (num_workers * 4)
    ):
        """
        Initialize worker pool.

        Args:
            num_workers: Number of worker processes
            worker_init_fn: Function to initialize model in worker
            worker_process_fn: Function to process batch in worker
            worker_configs: Config dict for each worker (device, model_path, etc.)
            queue_size: Queue size per worker (0 = auto)
        """
        self.num_workers = num_workers
        self.worker_init_fn = worker_init_fn
        self.worker_process_fn = worker_process_fn
        self.worker_configs = worker_configs

        # Auto queue size: num_workers * 4 batches
        self.queue_size = queue_size if queue_size > 0 else num_workers * 4

        # Create queues
        self._input_queues: List[Queue] = []
        self._output_queue: Queue = None

        # Worker processes
        self._workers: List[Process] = []
        self._running = False

        # Statistics
        self._batches_submitted = 0
        self._batches_completed = 0

    def start(self):
        """Start all worker processes."""
        if self._running:
            logger.warning("WorkerPool already running")
            return

        logger.info(f"Starting WorkerPool with {self.num_workers} workers")

        # Create output queue (shared by all workers)
        self._output_queue = mp.Queue()

        # Create input queues and workers
        for i in range(self.num_workers):
            # Input queue for this worker
            input_queue = mp.Queue(maxsize=self.queue_size)
            self._input_queues.append(input_queue)

            # Get config for this worker
            config = self.worker_configs[i] if i < len(self.worker_configs) else {}

            # Create worker process
            worker = Process(
                target=InferenceWorker.run,
                args=(
                    i,  # worker_id
                    input_queue,
                    self._output_queue,
                    self.worker_init_fn,
                    self.worker_process_fn,
                    config,
                ),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        self._running = True
        logger.info(f"WorkerPool started: {self.num_workers} workers, queue_size={self.queue_size}")

    def submit(self, worker_id: int, work_item: WorkItem) -> bool:
        """
        Submit work to a specific worker.

        Args:
            worker_id: Target worker (0 to num_workers-1)
            work_item: Work to process

        Returns:
            True if submitted, False if queue full
        """
        if not self._running:
            logger.warning("Cannot submit - pool not running")
            return False

        if worker_id < 0 or worker_id >= self.num_workers:
            raise ValueError(f"Invalid worker_id: {worker_id}")

        try:
            self._input_queues[worker_id].put(work_item, timeout=1.0)
            self._batches_submitted += 1
            return True
        except Exception:
            logger.warning(f"Queue full for worker {worker_id}")
            return False

    def submit_batch(self, batch_id: int, frames: List, metadata: List[Dict]) -> int:
        """
        Submit a batch using round-robin distribution.

        Args:
            batch_id: Batch identifier
            frames: List of frames
            metadata: List of metadata dicts

        Returns:
            Worker ID that received the batch
        """
        worker_id = batch_id % self.num_workers
        work_item = WorkItem(batch_id=batch_id, frames=frames, metadata=metadata)
        self.submit(worker_id, work_item)
        return worker_id

    def results(self, timeout: float = 1.0) -> Iterator[WorkResult]:
        """
        Iterate over results from workers.

        Args:
            timeout: Timeout for each get operation

        Yields:
            WorkResult objects
        """
        while self._running or not self._output_queue.empty():
            try:
                result = self._output_queue.get(timeout=timeout)
                self._batches_completed += 1
                yield result
            except Empty:
                if not self._running:
                    # Drain remaining results
                    while not self._output_queue.empty():
                        try:
                            result = self._output_queue.get_nowait()
                            self._batches_completed += 1
                            yield result
                        except Empty:
                            break
                    break

    def get_result(self, timeout: float = 1.0) -> Optional[WorkResult]:
        """
        Get a single result.

        Args:
            timeout: How long to wait

        Returns:
            WorkResult or None if timeout
        """
        try:
            result = self._output_queue.get(timeout=timeout)
            self._batches_completed += 1
            return result
        except Empty:
            return None

    def stop(self, timeout: float = 10.0):
        """
        Stop all workers gracefully.

        Args:
            timeout: Time to wait for graceful shutdown before force kill
        """
        if not self._running:
            return

        logger.info("Stopping WorkerPool...")
        self._running = False

        # Send sentinel to each worker
        for i, queue in enumerate(self._input_queues):
            try:
                queue.put(None, timeout=1.0)
            except Exception:
                logger.warning(f"Could not send shutdown to worker {i}")

        # Wait for workers to finish
        deadline = time.time() + timeout
        for i, worker in enumerate(self._workers):
            remaining = max(0.1, deadline - time.time())
            worker.join(timeout=remaining)

            if worker.is_alive():
                logger.warning(f"Worker {i} did not exit gracefully, terminating")
                worker.terminate()
                worker.join(timeout=1.0)

                if worker.is_alive():
                    logger.error(f"Worker {i} could not be terminated, killing")
                    worker.kill()

        # Clear queues
        for queue in self._input_queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Empty:
                    break

        logger.info(
            f"WorkerPool stopped: submitted={self._batches_submitted}, "
            f"completed={self._batches_completed}"
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def output_queue(self) -> Queue:
        """Access to output queue for advanced use."""
        return self._output_queue

    @property
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "num_workers": self.num_workers,
            "queue_size": self.queue_size,
            "batches_submitted": self._batches_submitted,
            "batches_completed": self._batches_completed,
            "running": self._running,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    print("Testing WorkerPool...")

    # Mock worker functions
    def mock_init(device: str, model_path: str, **kwargs):
        print(f"  Mock init: device={device}, path={model_path}")
        return {"device": device, "path": model_path}

    def mock_process(model, frames, metadata):
        time.sleep(0.01)  # Simulate processing
        return [{"frame_num": m["frame_num"], "detected": True} for m in metadata]

    # Create pool
    pool = WorkerPool(
        num_workers=2,
        worker_init_fn=mock_init,
        worker_process_fn=mock_process,
        worker_configs=[
            {"device": "cuda:0", "model_path": "/path/model"},
            {"device": "cuda:1", "model_path": "/path/model"},
        ],
        queue_size=4,
    )

    pool.start()

    # Submit some work
    for i in range(10):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
        metadata = [{"frame_num": i * 4 + j} for j in range(4)]
        pool.submit_batch(i, frames, metadata)

    # Collect results
    time.sleep(0.5)  # Let workers process

    results = []
    while True:
        result = pool.get_result(timeout=0.5)
        if result is None:
            break
        results.append(result)
        print(f"  Got result: batch={result.batch_id}, worker={result.worker_id}, time={result.processing_time_ms:.1f}ms")

    pool.stop()

    print(f"\nCollected {len(results)} results")
    print(f"Pool stats: {pool.stats}")
    print("Test complete!")
