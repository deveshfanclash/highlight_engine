"""
Video Output Handler

Combines VideoWriter and FrameAnnotator for easy service integration.
Provides a simple interface for writing annotated video output.

Design:
- Wraps VideoWriter + FrameAnnotator
- Simple interface: write_frame(frame, result)
- Handles lifecycle (open/close)
- Works with original resolution frames
- Context manager support
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np

from core.video_writer import VideoWriter, VideoWriterConfig
from core.visualization.annotator import FrameAnnotator, AnnotatorConfig

logger = logging.getLogger(__name__)


@dataclass
class VideoOutputConfig:
    """Configuration for VideoOutputHandler."""

    # Output settings
    output_path: Union[str, Path]
    fps: float = 30.0
    codec: str = "mp4v"

    # Annotation settings
    annotate: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    box_thickness: int = 2
    font_scale: float = 0.5

    # Coordinate format from detection results
    # "normalized" = 0-1 range (default from ODService)
    coord_format: str = "normalized"

    def __post_init__(self):
        self.output_path = Path(self.output_path)


class VideoOutputHandler:
    """
    Handles video output with optional annotation.

    Combines VideoWriter and FrameAnnotator into a simple interface
    for services to write processed frames.

    Usage:
        # Create handler
        handler = VideoOutputHandler(VideoOutputConfig(
            output_path="output.mp4",
            fps=30,
            annotate=True,
        ))

        # Process frames
        for frame, result in process_video():
            handler.write_frame(frame, result)

        # Finalize (or use context manager)
        handler.close()

        # With context manager (recommended)
        with VideoOutputHandler(config) as handler:
            for frame, result in process_video():
                handler.write_frame(frame, result)
    """

    def __init__(self, config: VideoOutputConfig):
        self.config = config

        # Create video writer config
        writer_config = VideoWriterConfig(
            output_path=config.output_path,
            fps=config.fps,
            codec=config.codec,
            frame_size=None,  # Auto-detect from first frame
        )
        self._writer = VideoWriter(writer_config)

        # Create annotator if enabled
        self._annotator: Optional[FrameAnnotator] = None
        if config.annotate:
            annotator_config = AnnotatorConfig(
                show_labels=config.show_labels,
                show_confidence=config.show_confidence,
                box_thickness=config.box_thickness,
                font_scale=config.font_scale,
                coord_format=config.coord_format,
            )
            self._annotator = FrameAnnotator(annotator_config)

        self._frame_count = 0

    def write_frame(
        self,
        frame: np.ndarray,
        result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write a frame to the video, optionally annotating it.

        Args:
            frame: Original BGR frame (not modified)
            result: Detection result dict (optional if annotate=False)

        Returns:
            True if write successful
        """
        # Annotate if enabled and result provided
        if self._annotator and result:
            frame = self._annotator.annotate(frame, result)
        elif self._annotator and result is None:
            # Annotation enabled but no result - just copy frame
            frame = frame.copy()

        # Write to video
        success = self._writer.write(frame)

        if success:
            self._frame_count += 1

        return success

    def close(self) -> None:
        """
        Close the video handler and finalize the video file.

        Safe to call multiple times.
        """
        self._writer.release()
        logger.info(f"VideoOutputHandler closed: {self._frame_count} frames written")

    @property
    def is_open(self) -> bool:
        """Check if handler is open."""
        return self._writer.is_open

    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count

    @property
    def output_path(self) -> Path:
        """Output file path."""
        return self.config.output_path

    # Context manager support
    def __enter__(self) -> "VideoOutputHandler":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_video_output(
    output_path: Union[str, Path],
    fps: float = 30.0,
    annotate: bool = True,
    show_labels: bool = True,
    show_confidence: bool = True,
    box_thickness: int = 2,
    codec: str = "mp4v",
) -> VideoOutputHandler:
    """
    Factory function to create a VideoOutputHandler.

    Args:
        output_path: Path to output video file
        fps: Frames per second
        annotate: Whether to draw detection boxes
        show_labels: Show class names on boxes
        show_confidence: Show confidence scores
        box_thickness: Bounding box thickness
        codec: Video codec

    Returns:
        Configured VideoOutputHandler instance

    Example:
        # Simple usage
        handler = create_video_output("output.mp4", fps=30)

        # Without annotation (raw frames)
        handler = create_video_output("raw.mp4", fps=30, annotate=False)
    """
    config = VideoOutputConfig(
        output_path=output_path,
        fps=fps,
        annotate=annotate,
        show_labels=show_labels,
        show_confidence=show_confidence,
        box_thickness=box_thickness,
        codec=codec,
    )
    return VideoOutputHandler(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os
    import cv2

    logging.basicConfig(level=logging.INFO)

    print("Testing VideoOutputHandler...\n")

    # Test 1: Basic annotated video output
    print("1. Testing annotated video output...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "annotated.mp4")

        # Create handler
        handler = create_video_output(output_path, fps=30, annotate=True)

        # Generate test frames with detections
        for i in range(30):
            # Create test frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (50 + i, 50, 50)  # Varying background

            # Create detection that moves across frame
            x_offset = i / 30.0 * 0.5
            result = {
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.95,
                        "bbox": {
                            "x1": 0.1 + x_offset,
                            "y1": 0.2,
                            "x2": 0.3 + x_offset,
                            "y2": 0.8,
                        },
                    },
                ]
            }

            success = handler.write_frame(frame, result)
            assert success, f"Failed to write frame {i}"

        handler.close()

        # Verify output
        assert os.path.exists(output_path)
        cap = cv2.VideoCapture(output_path)
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
        cap.release()

    print("   PASSED\n")

    # Test 2: Context manager
    print("2. Testing context manager...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "context.mp4")

        with create_video_output(output_path, fps=24) as handler:
            for i in range(10):
                frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                result = {
                    "detections": [
                        {
                            "class_id": 1,
                            "class_name": "ball",
                            "confidence": 0.88,
                            "bbox": {"x1": 0.4, "y1": 0.4, "x2": 0.6, "y2": 0.6},
                        },
                    ]
                }
                handler.write_frame(frame, result)

            assert handler.frame_count == 10

        # Verify video is valid after context exit
        cap = cv2.VideoCapture(output_path)
        assert cap.isOpened()
        cap.release()

    print("   PASSED\n")

    # Test 3: No annotation (raw frames)
    print("3. Testing raw frame output (no annotation)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "raw.mp4")

        with create_video_output(output_path, fps=30, annotate=False) as handler:
            for i in range(15):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:, :, 1] = i * 15  # Green gradient

                # Result is ignored when annotate=False
                handler.write_frame(frame, {"detections": []})

            assert handler.frame_count == 15

        # Verify
        cap = cv2.VideoCapture(output_path)
        assert cap.isOpened()
        cap.release()

    print("   PASSED\n")

    # Test 4: Write frame without result
    print("4. Testing write without result...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "no_result.mp4")

        with create_video_output(output_path, fps=30) as handler:
            for i in range(5):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                # No result provided
                handler.write_frame(frame)

            assert handler.frame_count == 5

    print("   PASSED\n")

    # Test 5: Multiple detections
    print("5. Testing multiple detections...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "multi_det.mp4")

        with create_video_output(output_path, fps=30) as handler:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (80, 80, 80)

            result = {
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.95,
                        "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.25, "y2": 0.8},
                    },
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.89,
                        "bbox": {"x1": 0.3, "y1": 0.25, "x2": 0.45, "y2": 0.75},
                    },
                    {
                        "class_id": 1,
                        "class_name": "ball",
                        "confidence": 0.92,
                        "bbox": {"x1": 0.5, "y1": 0.6, "x2": 0.55, "y2": 0.65},
                    },
                    {
                        "class_id": 2,
                        "class_name": "goal",
                        "confidence": 0.78,
                        "bbox": {"x1": 0.7, "y1": 0.1, "x2": 0.95, "y2": 0.9},
                    },
                ]
            }

            handler.write_frame(frame, result)

    print("   PASSED\n")

    print("=" * 50)
    print("All VideoOutputHandler tests PASSED!")
