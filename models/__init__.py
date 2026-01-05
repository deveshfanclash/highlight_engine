"""
Models Module

Provides model wrappers for different architectures:
- YOLO (YOLOv8, v11, v12)
- RF-DETR (future)
- Custom models
"""

from models.base_model import BaseModel, ModelOutput, Detection
from models.yolo_model import YOLOModel

__all__ = [
    "BaseModel",
    "ModelOutput",
    "Detection",
    "YOLOModel",
]
