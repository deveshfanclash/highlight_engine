"""
Test Execution Modes

Tests the unified local mode (single-frame and batched) and distributed mode.
Uses mock services to avoid GPU/video dependencies.
"""

import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from unittest.mock import MagicMock, patch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.base_service import BaseService, ServiceConfig
from input_handlers import FrameInputPacket

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# MOCK SERVICE FOR TESTING
# =============================================================================

@dataclass
class MockServiceConfig(ServiceConfig):
    """Config for mock service."""
    process_time_ms: float = 10.0  # Simulated processing time


class MockService(BaseService):
    """
    Mock service for testing execution modes.

    Simulates frame processing without actual model inference.
    """

    def __init__(self, config: MockServiceConfig):
        super().__init__(config)
        self.mock_config = config
        self._process_frame_calls = 0
        self._process_batch_calls = 0
        self._batch_sizes = []

    def initialize(self) -> bool:
        logger.info("MockService initialized")
        return True

    @property
    def supports_batching(self) -> bool:
        return True

    @property
    def supports_multi_worker(self) -> bool:
        return False  # Keep it simple for this test

    def process_frame(self, frame_packet: FrameInputPacket) -> Optional[Dict[str, Any]]:
        """Process single frame."""
        self._process_frame_calls += 1
        # Simulate processing time
        time.sleep(self.mock_config.process_time_ms / 1000)
        return {
            "frame_number": frame_packet.sequence_number,
            "detections": [],
            "detection_count": 0,
        }

    def process_batch(self, frame_packets: List[FrameInputPacket]) -> List[Optional[Dict[str, Any]]]:
        """Process batch of frames."""
        self._process_batch_calls += 1
        self._batch_sizes.append(len(frame_packets))
        # Simulate batch processing (slightly more efficient)
        time.sleep(self.mock_config.process_time_ms / 1000 * 1.2)  # Only 20% more time for whole batch
        return [
            {
                "frame_number": pkt.sequence_number,
                "detections": [],
                "detection_count": 0,
            }
            for pkt in frame_packets
        ]

    def cleanup(self):
        logger.info(f"MockService cleanup - process_frame calls: {self._process_frame_calls}, "
                   f"process_batch calls: {self._process_batch_calls}")


# =============================================================================
# MOCK FRAME GENERATOR
# =============================================================================

def create_mock_frames(num_frames: int):
    """Generate mock frame packets."""
    for i in range(num_frames):
        yield FrameInputPacket(
            sequence_number=i,
            timestamp_ms=i * 33,  # ~30 FPS
            data=np.zeros((100, 100, 3), dtype=np.uint8),
            frame=np.zeros((100, 100, 3), dtype=np.uint8),
            width=100,
            height=100,
        )


# =============================================================================
# TESTS
# =============================================================================

def test_local_mode_single_frame():
    """Test local mode with batch_size=1 (single-frame processing)."""
    print("\n" + "="*60)
    print("TEST: Local Mode - Single Frame (batch_size=1)")
    print("="*60)

    config = MockServiceConfig(
        match_id="test_match",
        service_id="mock_service",
        input_source="/fake/path.mp4",
        inference_batch_size=1,  # Single frame mode
        num_workers=1,
        process_time_ms=5.0,
    )

    service = MockService(config)
    num_frames = 10

    # Mock the input handler and db writer
    with patch.object(service, '_input_handler') as mock_input:
        with patch.object(service, '_db_writer') as mock_writer:
            mock_input.iterate.return_value = create_mock_frames(num_frames)
            mock_input.initialize.return_value = True
            mock_writer.queue_item = MagicMock()
            mock_writer.write_item = MagicMock()
            mock_writer.stop = MagicMock()

            # Bypass setup since we're mocking
            service._initialized = True
            service._input_handler = mock_input
            service._db_writer = mock_writer

            # Run the service
            service._running = True
            service._start_time = __import__('datetime').datetime.utcnow()
            service._run_local(batch_size=1)

    # Verify
    print(f"  process_frame() calls: {service._process_frame_calls}")
    print(f"  process_batch() calls: {service._process_batch_calls}")
    print(f"  Frames processed: {service._frames_processed}")

    assert service._process_frame_calls == num_frames, \
        f"Expected {num_frames} process_frame calls, got {service._process_frame_calls}"
    assert service._process_batch_calls == 0, \
        f"Expected 0 process_batch calls, got {service._process_batch_calls}"
    assert service._frames_processed == num_frames, \
        f"Expected {num_frames} frames processed, got {service._frames_processed}"

    print("  ✓ PASSED: Single-frame mode uses process_frame() correctly")
    return True


def test_local_mode_batched():
    """Test local mode with batch_size>1 (batched processing)."""
    print("\n" + "="*60)
    print("TEST: Local Mode - Batched (batch_size=4)")
    print("="*60)

    batch_size = 4
    num_frames = 10  # Will result in 2 full batches + 1 partial batch of 2

    config = MockServiceConfig(
        match_id="test_match",
        service_id="mock_service",
        input_source="/fake/path.mp4",
        inference_batch_size=batch_size,
        num_workers=1,
        process_time_ms=5.0,
    )

    service = MockService(config)

    # Mock the input handler and db writer
    with patch.object(service, '_input_handler') as mock_input:
        with patch.object(service, '_db_writer') as mock_writer:
            mock_input.iterate.return_value = create_mock_frames(num_frames)
            mock_input.initialize.return_value = True
            mock_writer.queue_item = MagicMock()
            mock_writer.write_item = MagicMock()
            mock_writer.stop = MagicMock()

            # Bypass setup since we're mocking
            service._initialized = True
            service._input_handler = mock_input
            service._db_writer = mock_writer

            # Run the service
            service._running = True
            service._start_time = __import__('datetime').datetime.utcnow()
            service._run_local(batch_size=batch_size)

    # Verify
    expected_batches = 3  # 4 + 4 + 2
    print(f"  process_frame() calls: {service._process_frame_calls}")
    print(f"  process_batch() calls: {service._process_batch_calls}")
    print(f"  Batch sizes: {service._batch_sizes}")
    print(f"  Frames processed: {service._frames_processed}")

    assert service._process_frame_calls == 0, \
        f"Expected 0 process_frame calls, got {service._process_frame_calls}"
    assert service._process_batch_calls == expected_batches, \
        f"Expected {expected_batches} process_batch calls, got {service._process_batch_calls}"
    assert service._batch_sizes == [4, 4, 2], \
        f"Expected batch sizes [4, 4, 2], got {service._batch_sizes}"
    assert service._frames_processed == num_frames, \
        f"Expected {num_frames} frames processed, got {service._frames_processed}"

    print("  ✓ PASSED: Batched mode uses process_batch() correctly")
    return True


def test_mode_selection():
    """Test that run() selects the correct mode based on config."""
    print("\n" + "="*60)
    print("TEST: Mode Selection Logic")
    print("="*60)

    # Test 1: batch_size=1 should use local mode with single-frame
    config1 = MockServiceConfig(
        match_id="test",
        service_id="test",
        input_source="/fake.mp4",
        inference_batch_size=1,
        num_workers=1,
    )
    service1 = MockService(config1)

    with patch.object(service1, 'setup', return_value=True):
        with patch.object(service1, '_run_local') as mock_local:
            with patch.object(service1, '_run_distributed') as mock_distributed:
                service1.run()
                mock_local.assert_called_once_with(1)  # positional arg
                mock_distributed.assert_not_called()
                print("  ✓ batch_size=1, num_workers=1 → _run_local(1)")

    # Test 2: batch_size=8 should use local mode with batching
    config2 = MockServiceConfig(
        match_id="test",
        service_id="test",
        input_source="/fake.mp4",
        inference_batch_size=8,
        num_workers=1,
    )
    service2 = MockService(config2)

    with patch.object(service2, 'setup', return_value=True):
        with patch.object(service2, '_run_local') as mock_local:
            with patch.object(service2, '_run_distributed') as mock_distributed:
                service2.run()
                mock_local.assert_called_once_with(8)  # positional arg
                mock_distributed.assert_not_called()
                print("  ✓ batch_size=8, num_workers=1 → _run_local(8)")

    # Test 3: num_workers>1 but service doesn't support it → fallback to local
    config3 = MockServiceConfig(
        match_id="test",
        service_id="test",
        input_source="/fake.mp4",
        inference_batch_size=4,
        num_workers=4,  # Requesting multi-worker
    )
    service3 = MockService(config3)  # But MockService.supports_multi_worker = False

    with patch.object(service3, 'setup', return_value=True):
        with patch.object(service3, '_run_local') as mock_local:
            with patch.object(service3, '_run_distributed') as mock_distributed:
                service3.run()
                mock_local.assert_called_once_with(4)  # positional arg
                mock_distributed.assert_not_called()
                print("  ✓ num_workers=4 but unsupported → _run_local(4) with warning")

    print("  ✓ PASSED: Mode selection logic works correctly")
    return True


def test_performance_comparison():
    """Compare performance of single-frame vs batched mode."""
    print("\n" + "="*60)
    print("TEST: Performance Comparison")
    print("="*60)

    num_frames = 20
    process_time_ms = 10.0

    # Single-frame mode
    config1 = MockServiceConfig(
        match_id="test",
        service_id="test",
        input_source="/fake.mp4",
        inference_batch_size=1,
        num_workers=1,
        process_time_ms=process_time_ms,
    )
    service1 = MockService(config1)

    start = time.perf_counter()
    with patch.object(service1, '_input_handler') as mock_input:
        with patch.object(service1, '_db_writer') as mock_writer:
            mock_input.iterate.return_value = create_mock_frames(num_frames)
            mock_writer.queue_item = MagicMock()
            mock_writer.write_item = MagicMock()
            mock_writer.stop = MagicMock()
            service1._input_handler = mock_input
            service1._db_writer = mock_writer
            service1._running = True
            service1._start_time = __import__('datetime').datetime.utcnow()
            service1._run_local(batch_size=1)
    single_frame_time = time.perf_counter() - start

    # Batched mode
    config2 = MockServiceConfig(
        match_id="test",
        service_id="test",
        input_source="/fake.mp4",
        inference_batch_size=4,
        num_workers=1,
        process_time_ms=process_time_ms,
    )
    service2 = MockService(config2)

    start = time.perf_counter()
    with patch.object(service2, '_input_handler') as mock_input:
        with patch.object(service2, '_db_writer') as mock_writer:
            mock_input.iterate.return_value = create_mock_frames(num_frames)
            mock_writer.queue_item = MagicMock()
            mock_writer.write_item = MagicMock()
            mock_writer.stop = MagicMock()
            service2._input_handler = mock_input
            service2._db_writer = mock_writer
            service2._running = True
            service2._start_time = __import__('datetime').datetime.utcnow()
            service2._run_local(batch_size=4)
    batched_time = time.perf_counter() - start

    print(f"  Single-frame mode: {single_frame_time*1000:.1f}ms for {num_frames} frames")
    print(f"  Batched mode (4):  {batched_time*1000:.1f}ms for {num_frames} frames")
    print(f"  Speedup: {single_frame_time/batched_time:.2f}x")

    # Batched should be faster (our mock simulates GPU efficiency)
    assert batched_time < single_frame_time, \
        "Batched mode should be faster than single-frame mode"

    print("  ✓ PASSED: Batched mode is faster as expected")
    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXECUTION MODES TEST SUITE")
    print("Testing unified local mode (single-frame + batched)")
    print("="*60)

    tests = [
        test_local_mode_single_frame,
        test_local_mode_batched,
        test_mode_selection,
        test_performance_comparison,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    failed = len(results) - passed

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)
