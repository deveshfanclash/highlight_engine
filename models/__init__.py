"""
Models Module

Provides model wrappers for different architectures:
- YOLO (YOLOv8, v11, v12)
- RF-DETR (future)
- Custom models

Also provides ModelRegistry for centralized model management.
"""

from models.base_model import BaseModel, ModelOutput, Detection
from models.yolo_model import YOLOModel
from models.registry import (
    ModelRegistry,
    get_global_registry,
    init_global_registry,
)

__all__ = [
    "BaseModel",
    "ModelOutput",
    "Detection",
    "YOLOModel",
    "ModelRegistry",
    "get_global_registry",
    "init_global_registry",
]
