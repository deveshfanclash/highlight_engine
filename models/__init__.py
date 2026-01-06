"""
Models Module

Provides model wrappers for different architectures:
- YOLO (YOLOv8, v11, v12)
- RT-DETR
- Whisper (audio transcription)
- Custom models via registry

Usage:
    from models import create_model_from_config, ModelRegistry

    # Create model from config
    model = create_model_from_config(model_config, device="cuda:0")

    # Register custom architecture
    ModelRegistry.register("my_arch", MyLoader)
"""

from models.base_model import BaseModel, ModelOutput, Detection
from models.yolo_model import YOLOModel
from models.whisper_model import WhisperModel, TranscriptionOutput, TranscriptionSegment
from models.registry import (
    ModelRegistry,
    ModelLoader,
    create_model,
    create_model_from_config,
    load_game_models,
    load_service_models,
)

__all__ = [
    # Base
    "BaseModel",
    "ModelOutput",
    "Detection",
    # Models
    "YOLOModel",
    "WhisperModel",
    "TranscriptionOutput",
    "TranscriptionSegment",
    # Registry
    "ModelRegistry",
    "ModelLoader",
    "create_model",
    "create_model_from_config",
    "load_game_models",
    "load_service_models",
]
