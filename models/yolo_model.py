"""
YOLO Model Wrapper

Wrapper for Ultralytics YOLO models (v8, v11, v12).
Provides consistent interface matching BaseModel.
"""

import time
import logging
from typing import List, Optional, Dict, Tuple

import numpy as np

from models.base_model import BaseModel, ModelOutput, Detection

logger = logging.getLogger(__name__)


class YOLOModel(BaseModel):
    """
    Wrapper for Ultralytics YOLO models.

    Supports:
    - YOLOv8 (detection, segmentation, pose)
    - YOLOv11
    - YOLOv12

    Usage:
        model = YOLOModel(
            model_id="football_od_v2",
            device="cuda:0",
            class_mapping={0: "PERSON", 1: "BALL"}
        )
        model.load("/path/to/model.pt")
        outputs = model.predict([frame1, frame2], confidence_threshold=0.5)
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        class_mapping: Optional[Dict[int, str]] = None,
        half_precision: bool = False
    ):
        """
        Initialize YOLO model.

        Args:
            model_id: Unique identifier for this model
            device: Device to run on ("cpu", "cuda:0", etc.)
            class_mapping: Map from model class IDs to universal class names
            half_precision: Use FP16 inference (faster on GPU)
        """
        super().__init__(model_id, device, class_mapping)
        self.half_precision = half_precision
        self._model_class_names: Dict[int, str] = {}

    def load(self, model_path: str) -> bool:
        """
        Load YOLO model from file.

        Args:
            model_path: Path to .pt model file

        Returns:
            True if loaded successfully
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")

        try:
            logger.info(f"Loading YOLO model from {model_path}")
            self._model = YOLO(model_path)

            # Get class names from model
            if hasattr(self._model, 'names'):
                self._model_class_names = self._model.names
            else:
                self._model_class_names = {}

            self._loaded = True
            logger.info(f"YOLO model loaded: {len(self._model_class_names)} classes")
            return True

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    def get_class_names(self) -> Dict[int, str]:
        """Get model's class names"""
        return self._model_class_names

    def predict(
        self,
        frames: List[np.ndarray],
        confidence_threshold: float = 0.5,
        classes: Optional[List[int]] = None,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        verbose: bool = False
    ) -> List[ModelOutput]:
        """
        Run YOLO inference on frames.

        Args:
            frames: List of frames (BGR format, HWC)
            confidence_threshold: Minimum confidence for detections
            classes: Filter to these class IDs (model's native IDs)
            iou_threshold: IOU threshold for NMS
            max_detections: Maximum detections per image
            verbose: Print YOLO output

        Returns:
            List of ModelOutput, one per frame
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        # Run YOLO inference
        results = self._model.predict(
            frames,
            device=self.device,
            conf=confidence_threshold,
            iou=iou_threshold,
            classes=classes,
            max_det=max_detections,
            verbose=verbose,
            half=self.half_precision and 'cuda' in self.device,
        )

        inference_time = (time.time() - start_time) * 1000  # ms

        # Process results
        outputs = []
        for i, result in enumerate(results):
            detections = self._process_result(result, frames[i].shape)
            outputs.append(ModelOutput(
                detections=detections,
                inference_time_ms=inference_time / len(frames),  # Per-frame time
                model_id=self.model_id,
                input_shape=(frames[i].shape[0], frames[i].shape[1]),
            ))

        return outputs

    def _process_result(self, result, frame_shape: Tuple[int, int, int]) -> List[Detection]:
        """
        Process a single YOLO result into Detection objects.

        Args:
            result: YOLO result object
            frame_shape: Shape of input frame (H, W, C)

        Returns:
            List of Detection objects with normalized coordinates
        """
        detections = []
        height, width = frame_shape[:2]

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for i in range(len(boxes)):
            # Get box coordinates (xyxy format)
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            # Normalize coordinates to 0-1 range
            x1 = round(float(box[0]) / width, 6)
            y1 = round(float(box[1]) / height, 6)
            x2 = round(float(box[2]) / width, 6)
            y2 = round(float(box[3]) / height, 6)

            # Clip to valid range
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            # Get class name (universal if mapped, else model's)
            class_name = self.map_class_id(cls_id)

            # Get universal class ID if available
            universal_id = None
            try:
                from config.class_registry import get_class_id
                universal_id = get_class_id(class_name)
            except ImportError:
                pass

            detections.append(Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=round(conf, 4),
                bbox=(x1, y1, x2, y2),
                universal_class_id=universal_id,
            ))

        return detections

    def warmup(self, input_shape: Tuple[int, int, int] = (640, 640, 3)):
        """Warm up the model"""
        super().warmup(input_shape)
        logger.info(f"YOLO model {self.model_id} warmed up")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_yolo_model(
    model_id: str,
    model_path: str,
    device: str = "cpu",
    class_mapping: Optional[Dict[int, str]] = None,
    half_precision: bool = False,
    warmup: bool = True
) -> YOLOModel:
    """
    Factory function to create and load a YOLO model.

    Args:
        model_id: Unique identifier for this model
        model_path: Path to .pt model file
        device: Device to run on
        class_mapping: Map from model class IDs to universal class names
        half_precision: Use FP16 inference
        warmup: Run warmup inference after loading

    Returns:
        Loaded YOLOModel instance
    """
    model = YOLOModel(
        model_id=model_id,
        device=device,
        class_mapping=class_mapping,
        half_precision=half_precision
    )

    if not model.load(model_path):
        raise RuntimeError(f"Failed to load model from {model_path}")

    if warmup:
        model.warmup()

    return model
