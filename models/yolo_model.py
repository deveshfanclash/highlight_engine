"""
YOLO Model Wrapper

Simple wrapper for Ultralytics YOLO models.
Automatically handles class mapping using the class registry.
"""

import time
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np

from config.class_registry import (
    find_model_class_ids,
    normalize_class_name,
    create_class_mapping,
)

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result with normalized coordinates (0-1)."""
    class_id: int  # Model's native class ID
    class_name: str  # Canonical class name (e.g., "BALL", "PERSON")
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized

    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": round(self.bbox[0], 6),
                "y1": round(self.bbox[1], 6),
                "x2": round(self.bbox[2], 6),
                "y2": round(self.bbox[3], 6),
            }
        }


@dataclass
class ModelOutput:
    """Output from model inference."""
    detections: List[Detection]
    inference_time_ms: float = 0.0

    def to_dict_list(self) -> List[Dict]:
        return [d.to_dict() for d in self.detections]


class YOLOModel:
    """
    YOLO model wrapper with automatic class mapping.

    Usage:
        model = YOLOModel("cuda:0")
        model.load("/path/to/model.pt")

        # Get classes user wants to detect
        class_ids = model.get_class_ids_for_names(["ball", "person"])

        # Run inference
        outputs = model.predict([frame], classes=class_ids)
    """

    def __init__(self, device: str = "cpu", half_precision: bool = False):
        self.device = device
        self.half_precision = half_precision
        self._model = None
        self._model_classes: Dict[int, str] = {}  # Model's native classes
        self._class_mapping: Dict[int, str] = {}  # Model ID -> Canonical name
        self._loaded = False

    def load(self, model_path: str) -> bool:
        """Load YOLO model and auto-discover its classes."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics required: pip install ultralytics")

        try:
            logger.info(f"Loading YOLO model: {model_path}")
            self._model = YOLO(model_path)

            # Auto-discover model classes
            if hasattr(self._model, 'names') and self._model.names:
                self._model_classes = dict(self._model.names)
            else:
                self._model_classes = {}

            # Auto-create class mapping (model ID -> canonical name)
            self._class_mapping = create_class_mapping(self._model_classes)

            self._loaded = True
            logger.info(f"Model loaded with {len(self._model_classes)} classes")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_model_classes(self) -> Dict[int, str]:
        """Get model's native class mapping {id: name}."""
        return self._model_classes

    def get_class_ids_for_names(self, class_names: List[str]) -> List[int]:
        """
        Get model class IDs for user-requested class names.

        Args:
            class_names: User's requested classes ["ball", "person"]

        Returns:
            List of model class IDs, or empty list for all classes
        """
        if not class_names:
            return []  # Empty = all classes
        return find_model_class_ids(self._model_classes, class_names)

    def predict(
        self,
        frames: List[np.ndarray],
        confidence: float = 0.5,
        classes: Optional[List[int]] = None,
        iou: float = 0.45,
        max_det: int = 100,
    ) -> List[ModelOutput]:
        """
        Run inference on frames.

        Args:
            frames: List of BGR frames
            confidence: Minimum confidence threshold
            classes: Model class IDs to detect (None = all)
            iou: IOU threshold for NMS
            max_det: Maximum detections per frame

        Returns:
            List of ModelOutput, one per frame
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        start = time.time()

        results = self._model.predict(
            frames,
            device=self.device,
            conf=confidence,
            iou=iou,
            classes=classes if classes else None,
            max_det=max_det,
            verbose=False,
            half=self.half_precision and 'cuda' in self.device,
        )

        inference_ms = (time.time() - start) * 1000

        outputs = []
        for i, result in enumerate(results):
            detections = self._process_result(result, frames[i].shape)
            outputs.append(ModelOutput(
                detections=detections,
                inference_time_ms=inference_ms / len(frames),
            ))

        return outputs

    def _process_result(self, result, frame_shape: Tuple) -> List[Detection]:
        """Process YOLO result into Detection objects."""
        detections = []
        h, w = frame_shape[:2]

        if result.boxes is None:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            # Normalize bbox to 0-1
            x1 = max(0.0, min(1.0, float(box[0]) / w))
            y1 = max(0.0, min(1.0, float(box[1]) / h))
            x2 = max(0.0, min(1.0, float(box[2]) / w))
            y2 = max(0.0, min(1.0, float(box[3]) / h))

            # Get canonical class name
            class_name = self._class_mapping.get(
                cls_id,
                self._model_classes.get(cls_id, f"class_{cls_id}")
            )

            detections.append(Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=round(conf, 4),
                bbox=(x1, y1, x2, y2),
            ))

        return detections

    def warmup(self, size: Tuple[int, int, int] = (640, 640, 3)):
        """Warm up model with dummy inference."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        dummy = np.zeros(size, dtype=np.uint8)
        self.predict([dummy])
        logger.info("Model warmed up")

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def load_yolo_model(
    model_path: str,
    device: str = "cpu",
    half_precision: bool = False,
    warmup: bool = True
) -> YOLOModel:
    """
    Load a YOLO model.

    Args:
        model_path: Path to .pt file
        device: Device ("cpu", "cuda:0", etc.)
        half_precision: Use FP16
        warmup: Run warmup inference

    Returns:
        Loaded YOLOModel
    """
    model = YOLOModel(device=device, half_precision=half_precision)
    if not model.load(model_path):
        raise RuntimeError(f"Failed to load model: {model_path}")
    if warmup:
        model.warmup()
    return model
