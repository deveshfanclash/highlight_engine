"""
Video Writer

Simple, sync video writer for outputting processed frames.
Isolated from services - can be used standalone.

Design:
- Auto-initializes on first frame (detects frame size)
- Produces valid video at any point (proper codec)
- Context manager support for clean resource management
- Easily extendible for future async support
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoWriterConfig:
    """Configuration for VideoWriter."""

    output_path: Union[str, Path]
    fps: float = 30.0
    codec: str = "mp4v"  # mp4v produces valid video at any point
    frame_size: Optional[Tuple[int, int]] = None  # (width, height), None = auto from first frame

    def __post_init__(self):
        self.output_path = Path(self.output_path)
        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)


class VideoWriter:
    """
    Simple sync video writer.

    Features:
    - Auto-initializes on first frame write
    - Produces valid/playable video even if stopped mid-way
    - Context manager support
    - Frame count tracking

    Usage:
        # Basic usage
        writer = VideoWriter(VideoWriterConfig(output_path="output.mp4", fps=30))
        for frame in frames:
            writer.write(frame)
        writer.release()

        # Context manager (recommended)
        with VideoWriter(config) as writer:
            for frame in frames:
                writer.write(frame)
        # Auto-releases on exit

        # Auto-detect frame size from first frame
        config = VideoWriterConfig(output_path="out.mp4", fps=30)  # No frame_size
        writer = VideoWriter(config)
        writer.write(first_frame)  # Initializes with first_frame's dimensions
    """

    def __init__(self, config: VideoWriterConfig):
        self.config = config
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count = 0
        self._frame_size: Optional[Tuple[int, int]] = config.frame_size
        self._initialized = False

    def _initialize(self, frame: np.ndarray) -> bool:
        """
        Initialize the underlying cv2.VideoWriter.

        Called automatically on first write if not already initialized.

        Args:
            frame: First frame (used for size detection if not configured)

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        # Detect frame size from first frame if not configured
        if self._frame_size is None:
            h, w = frame.shape[:2]
            self._frame_size = (w, h)
            logger.debug(f"Auto-detected frame size: {self._frame_size}")

        # Create fourcc code
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)

        # Create writer
        self._writer = cv2.VideoWriter(
            str(self.config.output_path),
            fourcc,
            self.config.fps,
            self._frame_size,
        )

        if not self._writer.isOpened():
            logger.error(f"Failed to open video writer: {self.config.output_path}")
            return False

        self._initialized = True
        logger.info(
            f"VideoWriter initialized: {self.config.output_path} "
            f"({self._frame_size[0]}x{self._frame_size[1]} @ {self.config.fps}fps)"
        )
        return True

    def write(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the video.

        Auto-initializes on first call if needed.

        Args:
            frame: BGR frame (numpy array, HxWxC)

        Returns:
            True if write successful
        """
        # Initialize on first frame
        if not self._initialized:
            if not self._initialize(frame):
                return False

        # Validate frame
        if frame is None or frame.size == 0:
            logger.warning(f"Skipping empty frame at index {self._frame_count}")
            return False

        # Resize if frame doesn't match expected size
        h, w = frame.shape[:2]
        if (w, h) != self._frame_size:
            frame = cv2.resize(frame, self._frame_size)

        # Write frame
        self._writer.write(frame)
        self._frame_count += 1

        return True

    def release(self) -> None:
        """
        Release the video writer and finalize the video file.

        Safe to call multiple times.
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None

            if self._initialized:
                logger.info(
                    f"VideoWriter released: {self.config.output_path} "
                    f"({self._frame_count} frames)"
                )

        self._initialized = False

    @property
    def is_open(self) -> bool:
        """Check if writer is open and ready."""
        return self._initialized and self._writer is not None and self._writer.isOpened()

    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count

    @property
    def frame_size(self) -> Optional[Tuple[int, int]]:
        """Frame size (width, height). None if not yet initialized."""
        return self._frame_size

    @property
    def output_path(self) -> Path:
        """Output file path."""
        return self.config.output_path

    # Context manager support
    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.release()
        return False  # Don't suppress exceptions


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_video_writer(
    output_path: Union[str, Path],
    fps: float = 30.0,
    codec: str = "mp4v",
    frame_size: Optional[Tuple[int, int]] = None,
) -> VideoWriter:
    """
    Factory function to create a VideoWriter.

    Args:
        output_path: Path to output video file
        fps: Frames per second
        codec: Video codec (default: mp4v)
        frame_size: (width, height) or None for auto-detection

    Returns:
        VideoWriter instance (not yet initialized - will init on first write)
    """
    config = VideoWriterConfig(
        output_path=output_path,
        fps=fps,
        codec=codec,
        frame_size=frame_size,
    )
    return VideoWriter(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os

    logging.basicConfig(level=logging.INFO)

    print("Testing VideoWriter...\n")

    # Test 1: Basic write with auto-detection
    print("1. Testing basic write with auto frame size detection...")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_auto.mp4")

        with create_video_writer(output_path, fps=30) as writer:
            # Generate test frames
            for i in range(30):
                # Create gradient frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:, :, 0] = i * 8  # Blue gradient
                frame[:, :, 1] = 128
                frame[:, :, 2] = 255 - i * 8  # Red gradient

                success = writer.write(frame)
                assert success, f"Failed to write frame {i}"

            assert writer.frame_count == 30
            assert writer.frame_size == (640, 480)

        # Verify file exists and has content
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify video is readable
        cap = cv2.VideoCapture(output_path)
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
        cap.release()

    print("   PASSED\n")

    # Test 2: Explicit frame size
    print("2. Testing explicit frame size...")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_explicit.mp4")

        config = VideoWriterConfig(
            output_path=output_path,
            fps=24,
            frame_size=(320, 240),
        )

        writer = VideoWriter(config)

        # Write frame with different size - should be resized
        large_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write(large_frame)

        assert writer.frame_size == (320, 240)

        writer.release()

        # Verify
        cap = cv2.VideoCapture(output_path)
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 240
        cap.release()

    print("   PASSED\n")

    # Test 3: Early release (video should still be valid)
    print("3. Testing early release (video validity)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_early.mp4")

        writer = create_video_writer(output_path, fps=30)

        # Write only 5 frames then release
        for i in range(5):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()

        # Video should be valid and playable
        cap = cv2.VideoCapture(output_path)
        assert cap.isOpened(), "Video should be playable after early release"

        # Read all frames to verify
        read_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            read_count += 1

        cap.release()
        assert read_count == 5, f"Expected 5 frames, got {read_count}"

    print("   PASSED\n")

    # Test 4: Multiple release calls (should be safe)
    print("4. Testing multiple release calls...")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_multi_release.mp4")

        writer = create_video_writer(output_path, fps=30)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)

        writer.release()
        writer.release()  # Should not raise
        writer.release()  # Should not raise

    print("   PASSED\n")

    print("=" * 50)
    print("All VideoWriter tests PASSED!")
