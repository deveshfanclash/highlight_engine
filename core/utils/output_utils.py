"""
Typed Results

Structured result classes for inference outputs.
Provides type safety, serialization, and filtering methods.

Usage:
    # Create detections
    det = Detection(
        class_id=0,
        class_name="PERSON",
        confidence=0.95,
        bbox=BoundingBox(x1=0.1, y1=0.2, x2=0.3, y2=0.8)
    )

    # Create results
    results = DetectionResults(
        detections=[det],
        frame_number=100,
        inference_time_ms=15.5
    )

    # Filter and serialize
    high_conf = results.filter(min_confidence=0.8)
    json_str = results.to_json()
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class BoxFormat(Enum):
    """Bounding box coordinate formats."""
    XYXY = "xyxy"      # x1, y1, x2, y2 (top-left, bottom-right)
    XYWH = "xywh"      # x, y, width, height (top-left corner)
    CXCYWH = "cxcywh"  # center_x, center_y, width, height


@dataclass
class BoundingBox:
    """
    Bounding box with normalized coordinates (0-1 range).

    Attributes:
        x1, y1: Top-left corner (normalized)
        x2, y2: Bottom-right corner (normalized)
    """
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        # Ensure valid range
        self.x1 = max(0.0, min(1.0, self.x1))
        self.y1 = max(0.0, min(1.0, self.y1))
        self.x2 = max(0.0, min(1.0, self.x2))
        self.y2 = max(0.0, min(1.0, self.y2))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Return as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Return as (x, y, width, height)."""
        return (self.x1, self.y1, self.width, self.height)

    def to_cxcywh(self) -> Tuple[float, float, float, float]:
        """Return as (center_x, center_y, width, height)."""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert to pixel coordinates."""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height)
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": round(self.x1, 6),
            "y1": round(self.y1, 6),
            "x2": round(self.x2, 6),
            "y2": round(self.y2, 6)
        }

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> "BoundingBox":
        return cls(x1=cx - w/2, y1=cy - h/2, x2=cx + w/2, y2=cy + h/2)

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class Detection:
    """
    Single detection result.

    Attributes:
        class_id: Model's native class ID
        class_name: Canonical class name (e.g., "PERSON", "BALL")
        confidence: Detection confidence (0-1)
        bbox: Bounding box (normalized coordinates)
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        bbox_data = data["bbox"]
        bbox = BoundingBox(
            x1=bbox_data["x1"],
            y1=bbox_data["y1"],
            x2=bbox_data["x2"],
            y2=bbox_data["y2"]
        )
        return cls(
            class_id=data["class_id"],
            class_name=data["class_name"],
            confidence=data["confidence"],
            bbox=bbox
        )


@dataclass
class Keypoint:
    """
    Single keypoint for pose estimation.

    Attributes:
        x, y: Normalized coordinates (0-1)
        confidence: Keypoint visibility/confidence
        name: Optional keypoint name (e.g., "left_shoulder")
    """
    x: float
    y: float
    confidence: float
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "confidence": round(self.confidence, 4)
        }
        if self.name:
            d["name"] = self.name
        return d

    def to_pixels(self, width: int, height: int) -> Tuple[int, int]:
        """Convert to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))

    @property
    def is_visible(self) -> bool:
        """Check if keypoint is visible (confidence > 0.5)."""
        return self.confidence > 0.5


@dataclass
class KeypointSkeleton:
    """
    Full skeleton with multiple keypoints.

    Attributes:
        keypoints: List of keypoints
        bbox: Bounding box of the person (optional)
        confidence: Overall detection confidence
    """
    keypoints: List[Keypoint]
    bbox: Optional[BoundingBox] = None
    confidence: float = 0.0

    # Standard COCO keypoint names
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        """Get keypoint by name."""
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "confidence": round(self.confidence, 4)
        }
        if self.bbox:
            d["bbox"] = self.bbox.to_dict()
        return d

    @property
    def visible_keypoints(self) -> List[Keypoint]:
        """Get only visible keypoints."""
        return [kp for kp in self.keypoints if kp.is_visible]


@dataclass
class Results:
    """
    Base class for inference results.

    Attributes:
        frame_number: Frame index
        timestamp_ms: Frame timestamp in milliseconds
        inference_time_ms: Model inference time
        model_id: Optional model identifier
    """
    frame_number: int
    timestamp_ms: int = 0
    inference_time_ms: float = 0.0
    model_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "model_id": self.model_id
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class DetectionResults(Results):
    """
    Object detection results.

    Attributes:
        detections: List of Detection objects
    """
    detections: List[Detection] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Total number of detections."""
        return len(self.detections)

    def filter(
        self,
        classes: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        max_detections: Optional[int] = None,
        min_area: Optional[float] = None,
    ) -> "DetectionResults":
        """
        Filter detections.

        Args:
            classes: Keep only these class names
            min_confidence: Minimum confidence threshold
            max_detections: Maximum number of detections to keep
            min_area: Minimum bbox area (normalized, 0-1)

        Returns:
            New DetectionResults with filtered detections
        """
        filtered = self.detections.copy()

        if classes:
            classes_upper = [c.upper() for c in classes]
            filtered = [d for d in filtered if d.class_name.upper() in classes_upper]

        if min_confidence is not None:
            filtered = [d for d in filtered if d.confidence >= min_confidence]

        if min_area is not None:
            filtered = [d for d in filtered if d.bbox.area >= min_area]

        # Sort by confidence and limit
        filtered = sorted(filtered, key=lambda d: d.confidence, reverse=True)
        if max_detections is not None:
            filtered = filtered[:max_detections]

        return DetectionResults(
            frame_number=self.frame_number,
            timestamp_ms=self.timestamp_ms,
            inference_time_ms=self.inference_time_ms,
            model_id=self.model_id,
            detections=filtered
        )

    def count_by_class(self) -> Dict[str, int]:
        """Count detections per class."""
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    def get_class(self, class_name: str) -> List[Detection]:
        """Get all detections of a specific class."""
        return [d for d in self.detections if d.class_name.upper() == class_name.upper()]

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["detections"] = [d.to_dict() for d in self.detections]
        base["detection_count"] = self.count
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResults":
        detections = [Detection.from_dict(d) for d in data.get("detections", [])]
        return cls(
            frame_number=data["frame_number"],
            timestamp_ms=data.get("timestamp_ms", 0),
            inference_time_ms=data.get("inference_time_ms", 0.0),
            model_id=data.get("model_id"),
            detections=detections
        )

    def to_yolo_format(self, image_width: int, image_height: int) -> List[str]:
        """
        Convert to YOLO annotation format.

        Format: class_id center_x center_y width height (all normalized)

        Returns:
            List of YOLO format strings, one per detection
        """
        lines = []
        for det in self.detections:
            cx, cy, w, h = det.bbox.to_cxcywh()
            lines.append(f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines


@dataclass
class PoseResults(Results):
    """
    Pose estimation results.

    Attributes:
        skeletons: List of KeypointSkeleton objects
    """
    skeletons: List[KeypointSkeleton] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of detected people."""
        return len(self.skeletons)

    def filter(
        self,
        min_confidence: Optional[float] = None,
        min_visible_keypoints: Optional[int] = None,
    ) -> "PoseResults":
        """
        Filter skeletons.

        Args:
            min_confidence: Minimum detection confidence
            min_visible_keypoints: Minimum number of visible keypoints

        Returns:
            New PoseResults with filtered skeletons
        """
        filtered = self.skeletons.copy()

        if min_confidence is not None:
            filtered = [s for s in filtered if s.confidence >= min_confidence]

        if min_visible_keypoints is not None:
            filtered = [s for s in filtered if len(s.visible_keypoints) >= min_visible_keypoints]

        return PoseResults(
            frame_number=self.frame_number,
            timestamp_ms=self.timestamp_ms,
            inference_time_ms=self.inference_time_ms,
            model_id=self.model_id,
            skeletons=filtered
        )

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["skeletons"] = [s.to_dict() for s in self.skeletons]
        base["person_count"] = self.count
        return base


@dataclass
class ClassificationResult:
    """Single classification result."""
    class_id: int
    class_name: str
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "probability": round(self.probability, 4)
        }


@dataclass
class ClassificationResults(Results):
    """
    Classification results.

    Attributes:
        predictions: List of (class_id, class_name, probability) sorted by probability
    """
    predictions: List[ClassificationResult] = field(default_factory=list)

    @property
    def top_class(self) -> Optional[ClassificationResult]:
        """Get top prediction."""
        return self.predictions[0] if self.predictions else None

    def top_k(self, k: int = 5) -> List[ClassificationResult]:
        """Get top-k predictions."""
        return self.predictions[:k]

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["predictions"] = [p.to_dict() for p in self.predictions]
        if self.top_class:
            base["top_class"] = self.top_class.class_name
            base["top_confidence"] = self.top_class.probability
        return base


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_detection(
    class_id: int,
    class_name: str,
    confidence: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> Detection:
    """Create a Detection with xyxy coordinates."""
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    )


def create_detection_results(
    frame_number: int,
    detections: List[Dict[str, Any]],
    timestamp_ms: int = 0,
    inference_time_ms: float = 0.0,
    model_id: Optional[str] = None
) -> DetectionResults:
    """
    Create DetectionResults from list of detection dicts.

    Each dict should have: class_id, class_name, confidence, bbox (x1, y1, x2, y2)
    """
    det_objects = []
    for d in detections:
        bbox = d.get("bbox", {})
        det_objects.append(Detection(
            class_id=d["class_id"],
            class_name=d["class_name"],
            confidence=d["confidence"],
            bbox=BoundingBox(
                x1=bbox.get("x1", 0),
                y1=bbox.get("y1", 0),
                x2=bbox.get("x2", 0),
                y2=bbox.get("y2", 0)
            )
        ))

    return DetectionResults(
        frame_number=frame_number,
        timestamp_ms=timestamp_ms,
        inference_time_ms=inference_time_ms,
        model_id=model_id,
        detections=det_objects
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Typed Results...")

    # Test BoundingBox
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8)
    assert abs(bbox.width - 0.4) < 0.001
    assert abs(bbox.height - 0.6) < 0.001
    assert abs(bbox.area - 0.24) < 0.001
    print("  BoundingBox works")

    # Test format conversions
    xywh = bbox.to_xywh()
    assert abs(xywh[2] - 0.4) < 0.001  # width
    cxcywh = bbox.to_cxcywh()
    assert abs(cxcywh[0] - 0.3) < 0.001  # center_x
    print("  Box format conversions work")

    # Test Detection
    det = Detection(
        class_id=0,
        class_name="PERSON",
        confidence=0.95,
        bbox=bbox
    )
    det_dict = det.to_dict()
    assert det_dict["class_name"] == "PERSON"
    print("  Detection works")

    # Test DetectionResults
    results = DetectionResults(
        frame_number=100,
        timestamp_ms=3333,
        inference_time_ms=15.5,
        model_id="yolo_v8",
        detections=[
            det,
            Detection(class_id=1, class_name="BALL", confidence=0.8,
                     bbox=BoundingBox(0.4, 0.4, 0.5, 0.5)),
            Detection(class_id=0, class_name="PERSON", confidence=0.6,
                     bbox=BoundingBox(0.6, 0.2, 0.9, 0.9)),
        ]
    )

    assert results.count == 3
    assert results.count_by_class() == {"PERSON": 2, "BALL": 1}
    print("  DetectionResults basic works")

    # Test filtering
    filtered = results.filter(classes=["person"], min_confidence=0.7)
    assert filtered.count == 1
    assert filtered.detections[0].confidence == 0.95
    print("  DetectionResults filtering works")

    # Test serialization
    json_str = results.to_json()
    assert "PERSON" in json_str
    parsed = json.loads(json_str)
    assert parsed["detection_count"] == 3
    print("  Serialization works")

    # Test from_dict
    restored = DetectionResults.from_dict(parsed)
    assert restored.count == 3
    print("  Deserialization works")

    # Test YOLO format
    yolo_lines = results.to_yolo_format(1920, 1080)
    assert len(yolo_lines) == 3
    assert yolo_lines[0].startswith("0 ")  # class_id 0
    print("  YOLO format export works")

    # Test factory function
    results2 = create_detection_results(
        frame_number=1,
        detections=[
            {"class_id": 0, "class_name": "PERSON", "confidence": 0.9,
             "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5}}
        ]
    )
    assert results2.count == 1
    print("  Factory function works")

    print("\nAll tests passed!")
