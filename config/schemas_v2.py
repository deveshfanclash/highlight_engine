"""
Configuration Schemas v2 - Simplified and Extensible

Design Principles:
1. Clear separation: Input → Models → Services → Output
2. Registry pattern: Easy to add new architectures/services without code changes
3. Validation at load time, not runtime
4. Simple YAML configs that are easy to read/write

Layers:
- InputConfig: Stream source and frame extraction settings
- ModelConfig: Model definitions (weights, classes, architecture)
- ServiceConfig: What services to run and how
- OutputConfig: Where to write results
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from datetime import datetime
import re


# =============================================================================
# ENUMS - Keep minimal, use strings for extensibility
# =============================================================================

class GameCategory(str, Enum):
    """Game categories for grouping"""
    BALL_SPORT = "ball_sport"
    RACKET_SPORT = "racket_sport"
    COMBAT_SPORT = "combat_sport"
    ATHLETICS = "athletics"
    ESPORTS = "esports"
    OTHER = "other"


# =============================================================================
# INPUT LAYER - Stream and Frame Configuration
# =============================================================================

class InputConfig(BaseModel):
    """
    Configuration for input source and frame extraction.

    Supports: HLS, RTSP, MP4, and other video formats.
    """
    # Source
    source_type: Literal["hls", "rtsp", "mp4", "file", "rtmp"] = "hls"

    # Frame extraction
    resolution: Optional[List[int]] = Field(
        default=None,
        description="Target resolution [width, height]. None = use source."
    )
    fps: Optional[float] = Field(
        default=None,
        description="Target FPS. None = use source FPS."
    )
    frame_skip: int = Field(
        default=1,
        ge=1,
        description="Process every Nth frame. 1 = all frames."
    )

    # Resume support
    start_frame: int = Field(default=0, ge=0)
    start_segment: int = Field(default=1, ge=1)

    @field_validator('resolution')
    @classmethod
    def validate_resolution(cls, v):
        if v is not None:
            if len(v) != 2:
                raise ValueError("Resolution must be [width, height]")
            if v[0] <= 0 or v[1] <= 0:
                raise ValueError("Resolution must be positive")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "source_type": "hls",
                "resolution": [1280, 720],
                "frame_skip": 1
            }
        }


# =============================================================================
# MODEL LAYER - Architecture-agnostic model configuration
# =============================================================================

class ClassMapping(BaseModel):
    """Maps model's output class to universal class name"""
    model_class: int = Field(..., description="Class ID in model output")
    name: str = Field(..., description="Universal class name (e.g., PERSON, BALL)")
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Override confidence for this class"
    )


class ModelConfig(BaseModel):
    """
    Model configuration - architecture agnostic.

    The 'architecture' field determines how the model is loaded.
    New architectures can be added by registering a loader.

    Supported architectures (extensible via registry):
    - yolov8, yolov11, yolov12: Ultralytics YOLO models
    - rf_detr: RT-DETR models
    - segment_anything: SAM models
    - whisper: Audio transcription
    - gemini, openai: LLM APIs
    - custom: User-defined loader
    """
    # Identity
    id: str = Field(..., description="Unique model identifier")
    name: Optional[str] = Field(default=None, description="Human-readable name")

    # Architecture - string for extensibility
    architecture: str = Field(
        ...,
        description="Model architecture (yolov8, whisper, gemini, custom, etc.)"
    )

    # Weights
    weights: str = Field(
        ...,
        description="Path to weights (S3 URL, local path, or model ID)"
    )
    version: str = Field(default="latest")

    # Output configuration
    output_type: Literal["bbox", "mask", "keypoints", "text", "embedding", "classification"] = "bbox"

    # Class mapping (for detection/segmentation models)
    classes: List[ClassMapping] = Field(
        default_factory=list,
        description="Class mapping for detection models"
    )

    # Model parameters - architecture specific
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Architecture-specific parameters (confidence, iou, etc.)"
    )

    # Custom loader (for architecture="custom")
    custom_loader: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom loader config: {module, class, load_method}"
    )

    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        if not re.match(r'^[a-z0-9_]+$', v):
            raise ValueError("Model ID must be lowercase alphanumeric with underscores")
        return v

    @model_validator(mode='after')
    def validate_custom_loader(self):
        if self.architecture == "custom" and not self.custom_loader:
            raise ValueError("custom_loader required when architecture='custom'")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "id": "football_od_v2",
                "architecture": "yolov8",
                "weights": "s3://models/football/od_v2.pt",
                "output_type": "bbox",
                "classes": [
                    {"model_class": 0, "name": "PERSON"},
                    {"model_class": 1, "name": "BALL"}
                ],
                "params": {
                    "confidence": 0.5,
                    "iou": 0.45
                }
            }
        }


# =============================================================================
# SERVICE LAYER - What to run and how
# =============================================================================

class ServiceConfig(BaseModel):
    """
    Service configuration - defines what processing to run.

    Services are independent processing units that:
    - Read frames from the input source
    - Run one or more models
    - Write results to output

    Supported service types (extensible via registry):
    - object_detection: Run OD models, output bboxes
    - camera_view: Detect camera cuts
    - segmentation: Run segmentation models, output masks
    - replay_detection: Detect replay segments
    - transcription: Audio to text
    - event_detection: Detect game events
    """
    # Identity
    id: str = Field(..., description="Unique service identifier")
    type: str = Field(..., description="Service type (object_detection, camera_view, etc.)")
    enabled: bool = Field(default=True)

    # Resource allocation
    device: str = Field(
        default="cpu",
        description="Device to run on (cpu, cuda:0, cuda:1, etc.)"
    )

    # Models to use (for model-based services)
    models: List[str] = Field(
        default_factory=list,
        description="List of model IDs to run"
    )

    # Service-specific parameters
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Service-specific parameters"
    )

    # Input override (optional - defaults to global input config)
    input_override: Optional[InputConfig] = Field(
        default=None,
        description="Override input config for this service"
    )

    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        if not re.match(r'^(cpu|cuda:\d+|mps)$', v):
            raise ValueError(f"Invalid device format: {v}. Use 'cpu', 'cuda:0', etc.")
        return v

    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        if not re.match(r'^[a-z0-9_]+$', v):
            raise ValueError("Service ID must be lowercase alphanumeric with underscores")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "id": "od_football",
                "type": "object_detection",
                "device": "cuda:0",
                "models": ["football_od_v2"],
                "params": {}
            }
        }


# =============================================================================
# OUTPUT LAYER - Where to write results
# =============================================================================

class OutputConfig(BaseModel):
    """
    Output configuration - where results are written.

    Supports multiple output targets:
    - dynamodb: AWS DynamoDB table
    - s3: AWS S3 bucket (for large outputs like masks)
    - local: Local files (for testing)
    - kafka: Kafka topic (for streaming)
    """
    # Primary output (detections, events)
    primary: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "dynamodb", "table": "inference_results"},
        description="Primary output destination"
    )

    # Secondary output (masks, embeddings - large data)
    secondary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Secondary output for large data (S3)"
    )

    # Batching
    batch_size: int = Field(default=12, ge=1)
    flush_interval_ms: int = Field(default=250, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "primary": {"type": "dynamodb", "table": "inference_results"},
                "secondary": {"type": "s3", "bucket": "inference-outputs"},
                "batch_size": 12
            }
        }


# =============================================================================
# GAME CONFIG - Ties everything together
# =============================================================================

class GameConfig(BaseModel):
    """
    Complete game configuration.

    A game config defines:
    - What game/sport this is for
    - What models are available
    - What services to run
    - Input/output settings
    """
    # Identity
    id: str = Field(..., description="Unique game identifier")
    name: str = Field(..., description="Human-readable name")
    category: GameCategory = Field(default=GameCategory.BALL_SPORT)

    # Input settings
    input: InputConfig = Field(default_factory=InputConfig)

    # Available models
    models: List[ModelConfig] = Field(
        default_factory=list,
        description="Models available for this game"
    )

    # Services to run
    services: List[ServiceConfig] = Field(
        default_factory=list,
        description="Services to run for this game"
    )

    # Output settings
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Metadata
    version: int = Field(default=1)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @model_validator(mode='after')
    def validate_service_models(self):
        """Ensure all service model references exist"""
        model_ids = {m.id for m in self.models}
        for service in self.services:
            for model_id in service.models:
                if model_id not in model_ids:
                    raise ValueError(
                        f"Service '{service.id}' references unknown model '{model_id}'. "
                        f"Available models: {model_ids}"
                    )
        return self

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get model config by ID"""
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def get_service(self, service_id: str) -> Optional[ServiceConfig]:
        """Get service config by ID"""
        for service in self.services:
            if service.id == service_id:
                return service
        return None

    def get_enabled_services(self) -> List[ServiceConfig]:
        """Get only enabled services"""
        return [s for s in self.services if s.enabled]

    class Config:
        json_schema_extra = {
            "example": {
                "id": "football",
                "name": "Football",
                "category": "ball_sport",
                "models": [
                    {
                        "id": "football_od_v2",
                        "architecture": "yolov8",
                        "weights": "s3://models/football/od_v2.pt",
                        "classes": [
                            {"model_class": 0, "name": "PERSON"},
                            {"model_class": 1, "name": "BALL"}
                        ]
                    }
                ],
                "services": [
                    {
                        "id": "od_main",
                        "type": "object_detection",
                        "device": "cuda:0",
                        "models": ["football_od_v2"]
                    }
                ]
            }
        }


# =============================================================================
# MATCH CONFIG - Runtime configuration
# =============================================================================

class MatchStatus(str, Enum):
    """Match processing status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class MatchConfig(BaseModel):
    """
    Runtime configuration for a specific match.

    Contains:
    - Match identifier
    - Stream URL
    - Reference to game config (not a copy)
    - Runtime status
    """
    # Identity
    match_id: str = Field(..., description="Unique match identifier")
    game_id: str = Field(..., description="Reference to game config")

    # Stream
    stream_url: str = Field(..., description="Stream URL or file path")

    # Status
    status: MatchStatus = Field(default=MatchStatus.PENDING)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Resume support
    last_frame: int = Field(default=0, description="Last processed frame")
    last_segment: int = Field(default=1, description="Last processed segment")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (league, tournament, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "match_id": "match_20240105_001",
                "game_id": "football",
                "stream_url": "https://cdn.example.com/stream.m3u8",
                "status": "pending"
            }
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "GameCategory",
    "MatchStatus",
    # Config classes
    "InputConfig",
    "ClassMapping",
    "ModelConfig",
    "ServiceConfig",
    "OutputConfig",
    "GameConfig",
    "MatchConfig",
]
