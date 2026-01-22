"""
Models Module

Provides model wrappers for different architectures:
- YOLO (YOLOv8, v11, v12)

Also provides:
- ModelDownloader for caching and downloading from various sources
- ModelRegistry for centralized model management
"""

from models.yolo_model import YOLOModel, load_yolo_model, load_yolo_from_source
from models.downloader import (
    ModelDownloader,
    SourceType,
    DownloadResult,
    download_model,
    get_downloader,
)
from models.registry import (
    ModelRegistry,
    get_global_registry,
    init_global_registry,
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
    # Registry
    "ModelRegistry",
    "get_global_registry",
    "init_global_registry",
    # Results (convenience re-export)
    "Detection",
    "DetectionResults",
    "BoundingBox",
]
