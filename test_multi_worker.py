#!/usr/bin/env python3
"""
Test script for multi-worker support.

Tests:
1. WorkerPool with mock workers
2. FrameDistributor with mock pool
3. Integration test with mock data
"""

import numpy as np
import time
import sys
from dataclasses import dataclass


# =============================================================================
# MOCK FUNCTIONS (must be at module level for multiprocessing)
# =============================================================================

def mock_worker_init(device: str, model_path: str, **kwargs):
    """Mock model initialization."""
    print(f"  [Worker] Init: device={device}")
    return {"device": device, "path": model_path, "config": kwargs}


def mock_worker_process(model, frames, metadata):
    """Mock batch processing."""
    time.sleep(0.01)  # Simulate processing time
    results = []
    for i, meta in enumerate(metadata):
        results.append({
            "frame_num": meta.get("sequence_number", i),
            "detected": True,
            "detection_count": np.random.randint(0, 10),
        })
    return results


# =============================================================================
# TEST 1: WorkerPool
# =============================================================================

def test_worker_pool():
    """Test WorkerPool with mock workers."""
    from core.worker_pool import WorkerPool, WorkItem

    print("\n" + "=" * 60)
    print("TEST 1: WorkerPool")
    print("=" * 60)

    # Create pool with 2 workers
    pool = WorkerPool(
        num_workers=2,
        worker_init_fn=mock_worker_init,
        worker_process_fn=mock_worker_process,
        worker_configs=[
            {"device": "cpu", "model_path": "/mock/model.pt"},
            {"device": "cpu", "model_path": "/mock/model.pt"},
        ],
        queue_size=4,
    )

    print("Starting pool...")
    pool.start()
    time.sleep(0.5)  # Wait for workers to initialize

    # Submit some work
    print("Submitting 6 batches...")
    for i in range(6):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
        metadata = [{"sequence_number": i * 4 + j} for j in range(4)]
        pool.submit_batch(i, frames, metadata)

    # Collect results
    print("Collecting results...")
    time.sleep(1.0)  # Let workers process

    results = []
    timeout_start = time.time()
    while time.time() - timeout_start < 5.0:
        result = pool.get_result(timeout=0.5)
        if result is None:
            if len(results) >= 6:
                break
            continue
        results.append(result)
        print(f"  Got result: batch={result.batch_id}, worker={result.worker_id}, "
              f"time={result.processing_time_ms:.1f}ms")

    pool.stop()

    print(f"\nCollected {len(results)} results")
    print(f"Pool stats: {pool.stats}")

    success = len(results) == 6
    print(f"\nWorkerPool test: {'PASSED' if success else 'FAILED'}")
    return success


# =============================================================================
# TEST 2: FrameDistributor
# =============================================================================

@dataclass
class MockFramePacket:
    """Mock frame packet for testing."""
    frame: np.ndarray
    sequence_number: int
    timestamp_ms: int
    width: int = 100
    height: int = 100
    segment_number: int = 0


def test_frame_distributor():
    """Test FrameDistributor with mock pool."""
    from core.frame_distributor import FrameDistributor
    from core.worker_pool import WorkItem

    print("\n" + "=" * 60)
    print("TEST 2: FrameDistributor")
    print("=" * 60)

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

    # Create distributor
    pool = MockWorkerPool(num_workers=4)
    source = mock_frame_source(num_frames=32)

    distributor = FrameDistributor(
        frame_source=source,
        worker_pool=pool,
        batch_size=4,
    )

    print("Starting distributor...")
    distributor.start()
    distributor.wait(timeout=5.0)

    print(f"Frames distributed: {distributor.stats.frames_distributed}")
    print(f"Batches distributed: {distributor.stats.batches_distributed}")

    # Verify distribution
    print("\nDistribution by worker:")
    for worker_id in range(4):
        batches = pool._input_queues[worker_id]
        batch_ids = [b.batch_id for b in batches if b is not None]
        print(f"  Worker {worker_id}: batches {batch_ids}")

    # Verify batch distribution formula
    expected = {
        0: [0, 4],  # batch_id % 4 == 0
        1: [1, 5],  # batch_id % 4 == 1
        2: [2, 6],  # batch_id % 4 == 2
        3: [3, 7],  # batch_id % 4 == 3
    }

    success = True
    for worker_id, expected_batches in expected.items():
        actual = [b.batch_id for b in pool._input_queues[worker_id] if b is not None]
        if actual != expected_batches:
            print(f"  ERROR: Worker {worker_id} expected {expected_batches}, got {actual}")
            success = False

    print(f"\nFrameDistributor test: {'PASSED' if success else 'FAILED'}")
    return success


# =============================================================================
# TEST 3: Auto-Buffer Detection
# =============================================================================

def test_auto_buffer():
    """Test auto-buffer detection in FrameInputHandler."""
    from input_handlers.frame_handler import FrameInputHandler
    from config.schemas import InputType
    from core.stream_buffer import BufferMode

    print("\n" + "=" * 60)
    print("TEST 3: Auto-Buffer Detection")
    print("=" * 60)

    test_cases = [
        # (source, input_type, expected_buffered, expected_mode)
        ("https://example.com/stream.m3u8", InputType.HLS, True, BufferMode.DROP_OLD),
        ("rtsp://192.168.1.1/stream", InputType.RTSP, True, BufferMode.DROP_OLD),
        ("/path/to/video.mp4", InputType.MP4, False, BufferMode.FIFO),
        ("./local_file.avi", InputType.FILE, False, BufferMode.FIFO),
    ]

    success = True
    for source, input_type, expected_buffered, expected_mode in test_cases:
        handler = FrameInputHandler(
            input_source=source,
            input_type=input_type,
            buffered=None,  # Auto-detect
        )

        result_ok = (handler.buffered == expected_buffered and
                    handler.buffer_mode == expected_mode)

        status = "OK" if result_ok else "FAIL"
        print(f"  {status}: {source[:40]:40} -> buffered={handler.buffered}, mode={handler.buffer_mode.value}")

        if not result_ok:
            success = False

    print(f"\nAuto-Buffer test: {'PASSED' if success else 'FAILED'}")
    return success


# =============================================================================
# TEST 4: Config Settings
# =============================================================================

def test_config_settings():
    """Test that config settings are properly defined."""
    from config.schemas.game import InferenceSettings
    from services.base_service import ServiceConfig

    print("\n" + "=" * 60)
    print("TEST 4: Config Settings")
    print("=" * 60)

    # Test InferenceSettings defaults
    settings = InferenceSettings()
    print(f"InferenceSettings defaults:")
    print(f"  num_workers: {settings.num_workers}")
    print(f"  worker_queue_size: {settings.worker_queue_size}")
    print(f"  enable_buffering: {settings.enable_buffering}")
    print(f"  buffer_size: {settings.buffer_size}")
    print(f"  buffer_mode: {settings.buffer_mode}")

    # Verify defaults
    success = (
        settings.num_workers == 1 and
        settings.worker_queue_size == 0 and
        settings.enable_buffering is None and
        settings.buffer_size == 30 and
        settings.buffer_mode == "drop_old"
    )

    print(f"\nConfig Settings test: {'PASSED' if success else 'FAILED'}")
    return success


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("=" * 60)
    print("MULTI-WORKER SUPPORT TESTS")
    print("=" * 60)

    results = []

    # Test 1: WorkerPool (may fail due to multiprocessing in script)
    try:
        results.append(("WorkerPool", test_worker_pool()))
    except Exception as e:
        print(f"\nWorkerPool test ERROR: {e}")
        results.append(("WorkerPool", False))

    # Test 2: FrameDistributor
    try:
        results.append(("FrameDistributor", test_frame_distributor()))
    except Exception as e:
        print(f"\nFrameDistributor test ERROR: {e}")
        results.append(("FrameDistributor", False))

    # Test 3: Auto-Buffer
    try:
        results.append(("Auto-Buffer", test_auto_buffer()))
    except Exception as e:
        print(f"\nAuto-Buffer test ERROR: {e}")
        results.append(("Auto-Buffer", False))

    # Test 4: Config Settings
    try:
        results.append(("Config Settings", test_config_settings()))
    except Exception as e:
        print(f"\nConfig Settings test ERROR: {e}")
        results.append(("Config Settings", False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
