"""
Frame Annotator

Draws inference results (bounding boxes, labels) on frames.
Isolated from services - can be used standalone.

Design:
- Simple, focused on bounding box annotation
- Extendible for keypoints/masks in future
- Works with detection result dict format from services
- Configurable appearance (colors, thickness, fonts)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# COLOR PALETTE
# =============================================================================

# Default color palette (BGR format for OpenCV)
# Designed for good visibility on various backgrounds
DEFAULT_COLORS = [
    (255, 56, 56),    # Red
    (255, 157, 151),  # Light red
    (255, 112, 31),   # Orange
    (255, 178, 29),   # Yellow-orange
    (207, 210, 49),   # Yellow-green
    (72, 249, 10),    # Green
    (146, 204, 23),   # Light green
    (61, 219, 134),   # Teal
    (26, 147, 52),    # Dark green
    (0, 212, 187),    # Cyan
    (44, 153, 168),   # Dark cyan
    (0, 194, 255),    # Light blue
    (52, 69, 147),    # Dark blue
    (100, 115, 255),  # Blue
    (0, 24, 236),     # Deep blue
    (132, 56, 255),   # Purple
    (82, 0, 133),     # Dark purple
    (203, 56, 255),   # Magenta
    (255, 149, 200),  # Pink
    (255, 55, 199),   # Hot pink
]


def get_color_for_class(class_id: int, colors: List[Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
    """Get a consistent color for a class ID."""
    if colors is None:
        colors = DEFAULT_COLORS
    return colors[class_id % len(colors)]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AnnotatorConfig:
    """Configuration for FrameAnnotator."""

    # Box appearance
    box_thickness: int = 2
    box_color: Optional[Tuple[int, int, int]] = None  # None = color by class

    # Label appearance
    show_labels: bool = True
    show_confidence: bool = True
    font_scale: float = 0.5
    font_thickness: int = 1
    label_padding: int = 3
    label_bg_alpha: float = 0.7  # Background transparency (0-1)

    # Colors
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_COLORS)

    # Coordinate format
    # "normalized" = 0-1 range, "pixel" = absolute pixels
    coord_format: str = "normalized"


# =============================================================================
# FRAME ANNOTATOR
# =============================================================================

class FrameAnnotator:
    """
    Draws bounding boxes and labels on frames.

    Works with detection results in the format produced by ODService:
    {
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.5}
            },
            ...
        ]
    }

    Usage:
        annotator = FrameAnnotator()

        # Annotate with detection result dict
        annotated = annotator.annotate(frame, result_dict)

        # Or annotate with explicit detections list
        annotated = annotator.draw_detections(frame, detections)
    """

    def __init__(self, config: Optional[AnnotatorConfig] = None):
        self.config = config or AnnotatorConfig()

    def annotate(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
        class_names: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """
        Annotate frame with detection results.

        Args:
            frame: BGR frame (numpy array)
            result: Detection result dict with "detections" key
            class_names: Optional class name mapping {id: name}

        Returns:
            Annotated frame (copy of input)
        """
        # Make a copy to avoid modifying original
        annotated = frame.copy()

        # Extract detections
        detections = result.get("detections", [])
        if not detections:
            return annotated

        return self.draw_detections(annotated, detections, class_names)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        class_names: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels for detections.

        Args:
            frame: BGR frame (will be modified in-place)
            detections: List of detection dicts
            class_names: Optional class name mapping

        Returns:
            Annotated frame (same as input, modified in-place)
        """
        h, w = frame.shape[:2]

        for det in detections:
            # Extract bbox
            bbox = det.get("bbox", {})
            if not bbox:
                continue

            # Get coordinates
            x1 = bbox.get("x1", 0)
            y1 = bbox.get("y1", 0)
            x2 = bbox.get("x2", 0)
            y2 = bbox.get("y2", 0)

            # Convert normalized to pixel coordinates if needed
            if self.config.coord_format == "normalized":
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get class info
            class_id = det.get("class_id", 0)
            class_name = det.get("class_name", "")
            if not class_name and class_names:
                class_name = class_names.get(class_id, f"class_{class_id}")

            confidence = det.get("confidence", 0.0)

            # Get color
            if self.config.box_color:
                color = self.config.box_color
            else:
                color = get_color_for_class(class_id, self.config.colors)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                self.config.box_thickness,
            )

            # Draw label
            if self.config.show_labels or self.config.show_confidence:
                label = self._format_label(class_name, confidence)
                self._draw_label(frame, label, (x1, y1), color)

        return frame

    def _format_label(self, class_name: str, confidence: float) -> str:
        """Format label text."""
        parts = []

        if self.config.show_labels and class_name:
            parts.append(class_name)

        if self.config.show_confidence:
            parts.append(f"{confidence:.2f}")

        return " ".join(parts)

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw label with background at position."""
        if not label:
            return

        x, y = position
        padding = self.config.label_padding

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.config.font_thickness,
        )

        # Calculate background rectangle
        bg_x1 = x
        bg_y1 = y - text_h - 2 * padding
        bg_x2 = x + text_w + 2 * padding
        bg_y2 = y

        # Ensure label stays within frame
        h, w = frame.shape[:2]
        if bg_y1 < 0:
            # Move label below the box
            bg_y1 = y
            bg_y2 = y + text_h + 2 * padding

        bg_x2 = min(bg_x2, w)

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.addWeighted(
            overlay,
            self.config.label_bg_alpha,
            frame,
            1 - self.config.label_bg_alpha,
            0,
            frame,
        )

        # Draw text
        text_x = bg_x1 + padding
        text_y = bg_y2 - padding if bg_y1 < y else bg_y1 + text_h + padding

        # Use white or black text depending on background brightness
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            text_color,
            self.config.font_thickness,
            cv2.LINE_AA,
        )

    def draw_box(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw a single bounding box (utility method).

        Args:
            frame: BGR frame
            x1, y1, x2, y2: Box coordinates (pixels)
            color: BGR color
            thickness: Line thickness
            label: Optional label text

        Returns:
            Frame with box drawn
        """
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if label:
            self._draw_label(frame, label, (x1, y1), color)

        return frame


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_annotator(
    show_labels: bool = True,
    show_confidence: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.5,
    coord_format: str = "normalized",
) -> FrameAnnotator:
    """
    Factory function to create a FrameAnnotator.

    Args:
        show_labels: Show class names
        show_confidence: Show confidence scores
        box_thickness: Bounding box line thickness
        font_scale: Label font scale
        coord_format: "normalized" (0-1) or "pixel"

    Returns:
        Configured FrameAnnotator instance
    """
    config = AnnotatorConfig(
        show_labels=show_labels,
        show_confidence=show_confidence,
        box_thickness=box_thickness,
        font_scale=font_scale,
        coord_format=coord_format,
    )
    return FrameAnnotator(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os

    logging.basicConfig(level=logging.INFO)

    print("Testing FrameAnnotator...\n")

    # Test 1: Basic annotation
    print("1. Testing basic annotation...")

    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Dark gray background

    # Create test detections (normalized coordinates)
    result = {
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.8},
            },
            {
                "class_id": 1,
                "class_name": "ball",
                "confidence": 0.87,
                "bbox": {"x1": 0.5, "y1": 0.5, "x2": 0.6, "y2": 0.65},
            },
        ]
    }

    annotator = create_annotator()
    annotated = annotator.annotate(frame, result)

    # Verify output is different from input (boxes drawn)
    assert not np.array_equal(annotated, frame), "Frame should be modified"
    assert annotated.shape == frame.shape, "Shape should be preserved"

    print("   PASSED\n")

    # Test 2: Empty detections
    print("2. Testing empty detections...")

    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    result_empty = {"detections": []}

    annotated2 = annotator.annotate(frame2, result_empty)
    assert np.array_equal(annotated2, frame2), "Empty detections should not modify frame"

    print("   PASSED\n")

    # Test 3: Pixel coordinates
    print("3. Testing pixel coordinates...")

    config_pixel = AnnotatorConfig(coord_format="pixel")
    annotator_pixel = FrameAnnotator(config_pixel)

    frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
    result_pixel = {
        "detections": [
            {
                "class_id": 0,
                "class_name": "car",
                "confidence": 0.92,
                "bbox": {"x1": 100, "y1": 100, "x2": 300, "y2": 250},
            },
        ]
    }

    annotated3 = annotator_pixel.annotate(frame3, result_pixel)
    assert not np.array_equal(annotated3, frame3), "Frame should be modified"

    print("   PASSED\n")

    # Test 4: Custom colors
    print("4. Testing custom colors...")

    config_custom = AnnotatorConfig(
        box_color=(0, 255, 0),  # Green for all boxes
        show_confidence=False,
    )
    annotator_custom = FrameAnnotator(config_custom)

    frame4 = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated4 = annotator_custom.annotate(frame4, result)

    assert not np.array_equal(annotated4, frame4)

    print("   PASSED\n")

    # Test 5: Save annotated image
    print("5. Testing save annotated image...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "annotated.jpg")

        # Create more visible test frame
        frame5 = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        annotated5 = annotator.annotate(frame5, result)

        cv2.imwrite(output_path, annotated5)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify saved image is readable
        loaded = cv2.imread(output_path)
        assert loaded is not None
        assert loaded.shape == annotated5.shape

    print("   PASSED\n")

    # Test 6: Label at top edge (should move below)
    print("6. Testing label positioning at edges...")

    frame6 = np.zeros((480, 640, 3), dtype=np.uint8)
    result_edge = {
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "bbox": {"x1": 0.1, "y1": 0.0, "x2": 0.3, "y2": 0.2},  # At top edge
            },
        ]
    }

    annotated6 = annotator.annotate(frame6, result_edge)
    assert not np.array_equal(annotated6, frame6)

    print("   PASSED\n")

    print("=" * 50)
    print("All FrameAnnotator tests PASSED!")
