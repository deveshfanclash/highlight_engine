"""
Base Model

Abstract interface for all ML models used in inference.
Provides consistent API regardless of underlying model architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np


@dataclass
class Detection:
    """
    Single detection result.

    All coordinates are normalized (0-1 range).
    """
    class_id: int  # Model's native class ID
    class_name: str  # Universal class name (after mapping)
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized

    # Optional fields
    universal_class_id: Optional[int] = None  # Universal class ID (from registry)
    mask: Optional[np.ndarray] = None  # For segmentation models
    keypoints: Optional[List[Tuple[float, float, float]]] = None  # For pose models

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DB storage"""
        result = {
            "class_id": self.universal_class_id or self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": round(self.bbox[0], 6),
                "y1": round(self.bbox[1], 6),
                "x2": round(self.bbox[2], 6),
                "y2": round(self.bbox[3], 6),
            }
        }
        return result


@dataclass
class ModelOutput:
    """
    Output from a model's predict() call.

    Contains detections and metadata about the inference.
    """
    detections: List[Detection]
    inference_time_ms: float = 0.0
    model_id: str = ""

    # Additional metadata
    input_shape: Optional[Tuple[int, int]] = None  # (height, width)
    extra: Dict[str, Any] = field(default_factory=dict)

    def filter_by_confidence(self, min_confidence: float) -> "ModelOutput":
        """Return new ModelOutput with detections filtered by confidence"""
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return ModelOutput(
            detections=filtered,
            inference_time_ms=self.inference_time_ms,
            model_id=self.model_id,
            input_shape=self.input_shape,
            extra=self.extra,
        )

    def filter_by_class(self, class_names: List[str]) -> "ModelOutput":
        """Return new ModelOutput with detections filtered by class name"""
        filtered = [d for d in self.detections if d.class_name in class_names]
        return ModelOutput(
            detections=filtered,
            inference_time_ms=self.inference_time_ms,
            model_id=self.model_id,
            input_shape=self.input_shape,
            extra=self.extra,
        )

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert all detections to list of dicts"""
        return [d.to_dict() for d in self.detections]


class BaseModel(ABC):
    """
    Abstract base class for all ML models.

    Subclasses must implement:
    - load(): Load model weights
    - predict(): Run inference on frames
    - get_class_names(): Return model's class names

    Optional to implement:
    - warmup(): Warm up the model (run dummy inference)
    - preprocess(): Custom preprocessing
    - postprocess(): Custom postprocessing
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        class_mapping: Optional[Dict[int, str]] = None
    ):
        """
        Initialize model.

        Args:
            model_id: Unique identifier for this model
            device: Device to run on ("cpu", "cuda:0", etc.)
            class_mapping: Map from model class IDs to universal class names
        """
        self.model_id = model_id
        self.device = device
        self.class_mapping = class_mapping or {}
        self._model = None
        self._loaded = False

    @abstractmethod
    def load(self, model_path: str) -> bool:
        """
        Load model weights from file.

        Args:
            model_path: Path to model file

        Returns:
            True if loaded successfully
        """
        pass

    @abstractmethod
    def predict(
        self,
        frames: List[np.ndarray],
        confidence_threshold: float = 0.5,
        classes: Optional[List[int]] = None
    ) -> List[ModelOutput]:
        """
        Run inference on frames.

        Args:
            frames: List of frames (BGR format, HWC)
            confidence_threshold: Minimum confidence for detections
            classes: Filter to these class IDs (model's native IDs)

        Returns:
            List of ModelOutput, one per frame
        """
        pass

    @abstractmethod
    def get_class_names(self) -> Dict[int, str]:
        """
        Get model's class names.

        Returns:
            Dict mapping class ID to class name
        """
        pass

    def warmup(self, input_shape: Tuple[int, int, int] = (640, 640, 3)):
        """
        Warm up the model with a dummy inference.

        Args:
            input_shape: Shape of dummy input (H, W, C)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        dummy_frame = np.zeros(input_shape, dtype=np.uint8)
        self.predict([dummy_frame])

    def map_class_id(self, model_class_id: int) -> str:
        """
        Map model's class ID to universal class name.

        Args:
            model_class_id: Class ID from model output

        Returns:
            Universal class name, or model's class name if no mapping
        """
        if model_class_id in self.class_mapping:
            return self.class_mapping[model_class_id]

        # Fall back to model's class names
        model_classes = self.get_class_names()
        if model_class_id in model_classes:
            return model_classes[model_class_id]

        return f"class_{model_class_id}"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id}, device={self.device})"
