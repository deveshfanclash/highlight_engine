"""
Models Module

Provides model wrappers for different architectures:
- YOLO (YOLOv8, v11, v12)

Also provides:
- download_model for downloading from URLs, S3, HuggingFace
"""

from models.yolo_model import YOLOModel, load_yolo_model
from models.downloader import SourceType, download_model

# Re-export typed results for convenience
from core.utils.output_utils import Detection, DetectionResults, BoundingBox

__all__ = [
    # Model classes
    "YOLOModel",
    "load_yolo_model",
    # Downloader
    "SourceType",
    "download_model",
    # Results (convenience re-export)
    "Detection",
    "DetectionResults",
    "BoundingBox",
]
