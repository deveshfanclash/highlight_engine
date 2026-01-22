"""
Models Module

Provides model wrappers for different architectures:
- YOLO (YOLOv8, v11, v12)

Also provides:
- ModelDownloader for caching and downloading from various sources
"""

from models.yolo_model import YOLOModel, load_yolo_model, load_yolo_from_source
from models.downloader import (
    ModelDownloader,
    SourceType,
    DownloadResult,
    download_model,
    get_downloader,
)

# Re-export typed results for convenience
from output.results import Detection, DetectionResults, BoundingBox

__all__ = [
    # Model classes
    "YOLOModel",
    "load_yolo_model",
    "load_yolo_from_source",
    # Downloader
    "ModelDownloader",
    "SourceType",
    "DownloadResult",
    "download_model",
    "get_downloader",
    # Results (convenience re-export)
    "Detection",
    "DetectionResults",
    "BoundingBox",
]
