"""
Models Module

Provides model wrappers for different architectures:
- YOLO (YOLOv8, v11, v12)

Also provides ModelRegistry for centralized model management.
"""

from models.yolo_model import YOLOModel, Detection, ModelOutput, load_yolo_model
from models.registry import (
    ModelRegistry,
    get_global_registry,
    init_global_registry,
)

__all__ = [
    "Detection",
    "ModelOutput",
    "YOLOModel",
    "load_yolo_model",
    "ModelRegistry",
    "get_global_registry",
    "init_global_registry",
]
