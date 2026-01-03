"""
Configuration Schemas for Inference 2.0

Hierarchy:
- UniversalClass: Global class registry (PERSON, BALL, etc.)
- ModelConfig: Individual model definition (url, classes, params)
- ServiceConfig: Service definition (enabled, device, params)
- GameConfig: Per-game configuration (models, services, settings)
- MatchConfig: Runtime config for a specific match (frozen snapshot)
"""

from typing import Dict, List, Optional, Literal, Union, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime


# =============================================================================
# ENUMS
# =============================================================================

class ModelType(str, Enum):
    """Supported model types"""
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    POSE_ESTIMATION = "pose_estimation"
    CLASSIFICATION = "classification"


class ModelArchitecture(str, Enum):
    """Supported model architectures"""
    YOLO_V8 = "yolov8"
    YOLO_V11 = "yolov11"
    YOLO_V12 = "yolov12"
    RF_DETR = "rf_detr"
    CUSTOM = "custom"  # For custom model loading


class DeviceType(str, Enum):
    """Device types for running services"""
    CPU = "cpu"
    GPU = "gpu"  # Will use cuda:0 by default, or specify cuda:N


class ServiceType(str, Enum):
    """Available service types"""
    OBJECT_DETECTION = "object_detection"
    CAMERA_VIEW = "camera_view"
    REPLAY_DETECTION = "replay_detection"
    SEGMENTATION = "segmentation"
    POSE_ESTIMATION = "pose_estimation"


class OutputFormat(str, Enum):
    """Output format for detections"""
    BBOX = "bbox"  # Bounding box only (x1, y1, x2, y2 normalized)
    BBOX_WITH_MASK = "bbox_with_mask"  # Bbox + S3 path to mask (future)
    KEYPOINTS = "keypoints"  # For pose estimation


class GameCategory(str, Enum):
    """Game categories for grouping similar sports"""
    BALL_SPORT = "ball_sport"  # Football, Cricket, Volleyball
    RACKET_SPORT = "racket_sport"  # Tennis, Badminton, Table Tennis
    COMBAT_SPORT = "combat_sport"  # Boxing, MMA
    ATHLETICS = "athletics"  # Track and field
    OTHER = "other"


# =============================================================================
# UNIVERSAL CLASS REGISTRY
# =============================================================================

class UniversalClass(BaseModel):
    """
    Universal class definition - consistent across all games/models.
    Stored in MongoDB collection: universal_class_registry

    Example:
        {"class_id": 1, "class_name": "PERSON", "description": "Human player"}
    """
    class_id: int = Field(..., description="Universal class ID (consistent across all models)")
    class_name: str = Field(..., description="Universal class name (e.g., PERSON, BALL)")
    description: Optional[str] = Field(None, description="Human-readable description")
    applicable_sports: List[str] = Field(
        default=["all"],
        description="Sports where this class applies. Use 'all' for universal classes."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "class_id": 1,
                "class_name": "PERSON",
                "description": "Human player or person on field",
                "applicable_sports": ["all"]
            }
        }


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

class ModelParams(BaseModel):
    """
    Model-specific inference parameters.
    These are passed directly to the model's predict() method.
    """
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Minimum confidence for detections"
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0, le=1.0,
        description="IOU threshold for NMS"
    )
    max_detections: int = Field(
        default=100,
        ge=1,
        description="Maximum detections per frame"
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Inference batch size"
    )
    input_size: Optional[List[int]] = Field(
        default=None,
        description="Input resolution [width, height]. None = use model default."
    )
    half_precision: bool = Field(
        default=False,
        description="Use FP16 inference (faster on GPU)"
    )

    # Additional model-specific params can be added here
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model-specific parameters"
    )


class CustomModelConfig(BaseModel):
    """
    Configuration for custom model architectures.
    Used when model_architecture = "custom"
    """
    module_path: str = Field(
        ...,
        description="Python module path for custom model class (e.g., 'models.custom_yolo.CustomYOLO')"
    )
    class_name: str = Field(
        ...,
        description="Class name within the module"
    )
    init_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters passed to model __init__"
    )
    load_method: str = Field(
        default="load",
        description="Method name to call for loading weights"
    )
    predict_method: str = Field(
        default="predict",
        description="Method name to call for inference"
    )


class ClassMapping(BaseModel):
    """
    Maps model's native class IDs to universal class names.

    Example: Model outputs class 0 for person, but universal ID for PERSON is 1
    """
    model_class_id: int = Field(..., description="Class ID in model's output")
    universal_class_name: str = Field(..., description="Universal class name (e.g., PERSON)")

    # Optional: filter by confidence for this specific class
    class_confidence_threshold: Optional[float] = Field(
        default=None,
        description="Override confidence threshold for this class"
    )


class ModelConfig(BaseModel):
    """
    Complete configuration for a single model.

    A game can have multiple models (e.g., Cricket has 3).
    Each model runs as a separate service instance.
    """
    model_id: str = Field(
        ...,
        description="Unique identifier for this model (e.g., 'football_od_v2', 'cricket_person_head')"
    )
    model_type: ModelType = Field(
        ...,
        description="Type of model (object_detection, segmentation, etc.)"
    )
    model_architecture: ModelArchitecture = Field(
        default=ModelArchitecture.YOLO_V8,
        description="Model architecture"
    )
    model_url: str = Field(
        ...,
        description="URL to download model weights (S3, CDN, etc.)"
    )
    version: str = Field(
        default="latest",
        description="Model version (e.g., '2.1.0', 'latest')"
    )

    # Class configuration
    classes_to_predict: List[int] = Field(
        ...,
        description="Model's native class IDs to predict (filter others)"
    )
    class_mapping: List[ClassMapping] = Field(
        ...,
        description="Mapping from model class IDs to universal class names"
    )

    # Inference parameters
    params: ModelParams = Field(
        default_factory=ModelParams,
        description="Model inference parameters"
    )

    # Output format
    output_format: OutputFormat = Field(
        default=OutputFormat.BBOX,
        description="Output format for this model's detections"
    )

    # Custom model support
    custom_config: Optional[CustomModelConfig] = Field(
        default=None,
        description="Configuration for custom model architectures. Required if architecture='custom'"
    )

    # Device preference (can be overridden at service level)
    preferred_device: DeviceType = Field(
        default=DeviceType.GPU,
        description="Preferred device for this model"
    )

    @field_validator('custom_config')
    @classmethod
    def validate_custom_config(cls, v, info):
        """Ensure custom_config is provided when architecture is custom"""
        if info.data.get('model_architecture') == ModelArchitecture.CUSTOM and v is None:
            raise ValueError("custom_config is required when model_architecture is 'custom'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "football_od_v2",
                "model_type": "object_detection",
                "model_architecture": "yolov8",
                "model_url": "s3://spectatr-models/football/od_v2/latest.pt",
                "version": "2.1.0",
                "classes_to_predict": [0, 1, 2],
                "class_mapping": [
                    {"model_class_id": 0, "universal_class_name": "PERSON"},
                    {"model_class_id": 1, "universal_class_name": "BALL"},
                    {"model_class_id": 2, "universal_class_name": "GOAL"}
                ],
                "params": {
                    "confidence_threshold": 0.5,
                    "batch_size": 4
                },
                "output_format": "bbox",
                "preferred_device": "gpu"
            }
        }


# =============================================================================
# SERVICE CONFIGURATION
# =============================================================================

class ServiceParams(BaseModel):
    """
    Base class for service-specific parameters.
    Extended by specific service param classes.
    """
    pass


class ODServiceParams(ServiceParams):
    """Parameters specific to Object Detection service"""
    # Models are referenced by model_id from GameConfig.models
    model_ids: List[str] = Field(
        ...,
        description="List of model_ids to run in this service"
    )
    # If multiple models, each can be on different device
    device_assignment: Optional[Dict[str, str]] = Field(
        default=None,
        description="Map model_id to device (e.g., {'model_a': 'cuda:0', 'model_b': 'cuda:1'})"
    )


class CameraViewServiceParams(ServiceParams):
    """Parameters specific to Camera View Detection service"""
    phash_threshold: int = Field(
        default=20,
        description="Perceptual hash difference threshold for cut detection"
    )
    histogram_threshold: float = Field(
        default=0.90,
        description="Histogram correlation threshold (below = cut detected)"
    )
    min_frame_gap: int = Field(
        default=25,
        description="Minimum frames between camera cuts (prevents false positives)"
    )
    resolution_scale: float = Field(
        default=0.5,
        description="Scale factor for processing resolution (0.5 = half resolution)"
    )


class ReplayDetectionServiceParams(ServiceParams):
    """Parameters specific to Replay Detection service (future)"""
    # Placeholder for future implementation
    pass


class ServiceConfig(BaseModel):
    """
    Configuration for a single service.

    Services are independent units that:
    - Have their own frame provider (FFmpeg process)
    - Run on specified device (CPU/GPU)
    - Write to DynamoDB with their own PK (match_id#service_id)
    """
    service_type: ServiceType = Field(
        ...,
        description="Type of service"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this service should run"
    )
    device: str = Field(
        default="cpu",
        description="Device to run on ('cpu', 'cuda:0', 'cuda:1', etc.)"
    )

    # Service-specific parameters
    params: Union[ODServiceParams, CameraViewServiceParams, ReplayDetectionServiceParams, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Service-specific parameters"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "service_type": "object_detection",
                "enabled": True,
                "device": "cuda:0",
                "params": {
                    "model_ids": ["football_od_v2"]
                }
            }
        }


# =============================================================================
# INFERENCE SETTINGS
# =============================================================================

class InferenceSettings(BaseModel):
    """
    Global inference settings applied to all services.
    """
    target_fps: int = Field(
        default=25,
        description="Target FPS for frame extraction"
    )
    frame_skip: int = Field(
        default=1,
        description="Process every Nth frame (1 = all frames, 2 = every other frame)"
    )
    processing_resolution: List[int] = Field(
        default=[1280, 720],
        description="Resolution for processing [width, height]"
    )
    stream_buffer_size: int = Field(
        default=30,
        description="Buffer size for HLS stream segments"
    )
    db_write_batch_size: int = Field(
        default=12,
        description="Number of frames to batch before DB write"
    )
    db_write_interval_ms: int = Field(
        default=250,
        description="Maximum interval between DB writes in milliseconds"
    )


# =============================================================================
# GAME CONFIGURATION (Main Config)
# =============================================================================

class GameConfig(BaseModel):
    """
    Complete configuration for a game/sport.

    Stored in MongoDB collection: game_configs

    This is the main configuration that defines:
    - What models to use
    - What services to run
    - How to process the stream
    """
    # Identifiers
    game_id: str = Field(
        ...,
        description="Unique game identifier (MongoDB ObjectId as string)"
    )
    game_name: str = Field(
        ...,
        description="Human-readable game name (e.g., 'Football', 'Cricket')"
    )
    game_category: GameCategory = Field(
        default=GameCategory.BALL_SPORT,
        description="Category for grouping similar sports"
    )

    # Models - can have multiple models per game
    models: List[ModelConfig] = Field(
        ...,
        description="List of models for this game"
    )

    # Services - what services to run
    services: List[ServiceConfig] = Field(
        ...,
        description="List of services to run for this game"
    )

    # Global inference settings
    inference_settings: InferenceSettings = Field(
        default_factory=InferenceSettings,
        description="Global inference settings"
    )

    # Metadata
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)
    config_version: int = Field(
        default=1,
        description="Config version for tracking changes"
    )

    def get_model_by_id(self, model_id: str) -> Optional[ModelConfig]:
        """Helper to get model config by ID"""
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None

    def get_enabled_services(self) -> List[ServiceConfig]:
        """Helper to get only enabled services"""
        return [s for s in self.services if s.enabled]

    def get_service_by_type(self, service_type: ServiceType) -> Optional[ServiceConfig]:
        """Helper to get service config by type"""
        for service in self.services:
            if service.service_type == service_type:
                return service
        return None


# =============================================================================
# MATCH CONFIGURATION (Runtime)
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

    Stored in MongoDB collection: matches

    Contains:
    - Match identifiers
    - Stream information
    - Frozen config snapshot (from GameConfig at match start)
    - Runtime status
    """
    match_id: str = Field(
        ...,
        description="Unique match identifier"
    )
    game_id: str = Field(
        ...,
        description="Reference to game_id in game_configs"
    )
    stream_url: str = Field(
        ...,
        description="HLS/RTSP/MP4 stream URL"
    )
    stream_type: Literal["hls", "rtsp", "mp4"] = Field(
        default="hls",
        description="Type of stream"
    )

    # Status tracking
    status: MatchStatus = Field(
        default=MatchStatus.PENDING,
        description="Current processing status"
    )
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    # Frozen config - snapshot of GameConfig at match start
    # This ensures config changes don't affect running matches
    config_snapshot: Optional[GameConfig] = Field(
        default=None,
        description="Frozen GameConfig snapshot at match start"
    )

    # Additional metadata
    league: Optional[str] = Field(default=None)
    tournament_id: Optional[str] = Field(default=None)
    tournament_name: Optional[str] = Field(default=None)

    # Resume support
    last_processed_frame: int = Field(
        default=0,
        description="Last successfully processed frame (for resume)"
    )


# =============================================================================
# OUTPUT SCHEMAS (What gets written to DynamoDB)
# =============================================================================

class BoundingBox(BaseModel):
    """Normalized bounding box (0-1 range)"""
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)


class Detection(BaseModel):
    """Single detection result"""
    class_name: str = Field(..., description="Universal class name")
    class_id: int = Field(..., description="Universal class ID")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox


class InferenceOutput(BaseModel):
    """
    Output schema for inference results.
    Written to DynamoDB table: inference_results

    PK: {match_id}#{service_id}
    SK: {frame_number}
    """
    # Keys (for DynamoDB)
    pk: str = Field(..., description="Partition key: match_id#service_id")
    sk: int = Field(..., description="Sort key: frame_number")

    # Identifiers
    match_id: str
    service_id: str = Field(..., description="Service identifier (e.g., 'od_football_v2')")
    model_id: str = Field(..., description="Model that produced these detections")
    frame_number: int
    timestamp_ms: int = Field(..., description="Frame timestamp in video (milliseconds)")

    # Detections
    detections: List[Detection] = Field(default_factory=list)

    # Metadata
    processing_time_ms: int = Field(..., description="Time to process this frame")
    written_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def create_pk(cls, match_id: str, service_id: str) -> str:
        return f"{match_id}#{service_id}"


class CameraViewOutput(BaseModel):
    """
    Output schema for camera view detection results.
    Written to DynamoDB table: inference_results (same table, different service_id)
    """
    pk: str
    sk: int

    match_id: str
    service_id: str = Field(default="camera_view")
    frame_number: int
    timestamp_ms: int

    # Camera view specific
    is_camera_cut: bool
    phash_diff: Optional[int] = None
    histogram_correlation: Optional[float] = None

    written_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_default_football_config() -> GameConfig:
    """Factory function to create default Football config"""
    return GameConfig(
        game_id="6622504e845d0e572cddc306",
        game_name="Football",
        game_category=GameCategory.BALL_SPORT,
        models=[
            ModelConfig(
                model_id="football_od_v2",
                model_type=ModelType.OBJECT_DETECTION,
                model_architecture=ModelArchitecture.YOLO_V8,
                model_url="s3://spectatr-models/football/6622504e845d0e572cddc306/latest.pt",
                version="latest",
                classes_to_predict=[0, 1, 2],
                class_mapping=[
                    ClassMapping(model_class_id=0, universal_class_name="PERSON"),
                    ClassMapping(model_class_id=1, universal_class_name="BALL"),
                    ClassMapping(model_class_id=2, universal_class_name="GOAL"),
                ],
                params=ModelParams(confidence_threshold=0.5, batch_size=4),
            )
        ],
        services=[
            ServiceConfig(
                service_type=ServiceType.OBJECT_DETECTION,
                enabled=True,
                device="cuda:0",
                params=ODServiceParams(model_ids=["football_od_v2"]),
            ),
            ServiceConfig(
                service_type=ServiceType.CAMERA_VIEW,
                enabled=True,
                device="cpu",
                params=CameraViewServiceParams(),
            ),
        ],
        inference_settings=InferenceSettings(
            target_fps=25,
            processing_resolution=[1280, 720],
        ),
    )


def create_default_cricket_config() -> GameConfig:
    """Factory function to create default Cricket config (multi-model)"""
    return GameConfig(
        game_id="68b7fd9f414719f5a368fae0",
        game_name="Cricket",
        game_category=GameCategory.BALL_SPORT,
        models=[
            # Model 1: Cricket-specific detection
            ModelConfig(
                model_id="cricket_od_v1",
                model_type=ModelType.OBJECT_DETECTION,
                model_architecture=ModelArchitecture.YOLO_V8,
                model_url="s3://spectatr-models/cricket/cricket_od/latest.pt",
                version="latest",
                classes_to_predict=[0, 1, 2, 3],
                class_mapping=[
                    ClassMapping(model_class_id=0, universal_class_name="BATSMAN"),
                    ClassMapping(model_class_id=1, universal_class_name="BOWLER"),
                    ClassMapping(model_class_id=2, universal_class_name="BALL"),
                    ClassMapping(model_class_id=3, universal_class_name="WICKET"),
                ],
                params=ModelParams(confidence_threshold=0.5, batch_size=1),
            ),
            # Model 2: Person + Head detection
            ModelConfig(
                model_id="cricket_person_head_v1",
                model_type=ModelType.OBJECT_DETECTION,
                model_architecture=ModelArchitecture.YOLO_V8,
                model_url="s3://spectatr-models/cricket/person_head/latest.pt",
                version="latest",
                classes_to_predict=[0, 1],
                class_mapping=[
                    ClassMapping(model_class_id=0, universal_class_name="PERSON"),
                    ClassMapping(model_class_id=1, universal_class_name="HEAD"),
                ],
                params=ModelParams(confidence_threshold=0.4, batch_size=1),
            ),
            # Model 3: Generic person detection (YOLO default)
            ModelConfig(
                model_id="cricket_person_generic_v1",
                model_type=ModelType.OBJECT_DETECTION,
                model_architecture=ModelArchitecture.YOLO_V8,
                model_url="s3://spectatr-models/default/yolov8n.pt",
                version="latest",
                classes_to_predict=[0],  # Only person class from COCO
                class_mapping=[
                    ClassMapping(model_class_id=0, universal_class_name="PERSON"),
                ],
                params=ModelParams(confidence_threshold=0.5, batch_size=1),
            ),
        ],
        services=[
            # Each model runs as separate OD service on different GPU
            ServiceConfig(
                service_type=ServiceType.OBJECT_DETECTION,
                enabled=True,
                device="cuda:0",
                params=ODServiceParams(
                    model_ids=["cricket_od_v1"],
                ),
            ),
            ServiceConfig(
                service_type=ServiceType.OBJECT_DETECTION,
                enabled=True,
                device="cuda:1",
                params=ODServiceParams(
                    model_ids=["cricket_person_head_v1"],
                ),
            ),
            ServiceConfig(
                service_type=ServiceType.OBJECT_DETECTION,
                enabled=True,
                device="cuda:2",
                params=ODServiceParams(
                    model_ids=["cricket_person_generic_v1"],
                ),
            ),
            ServiceConfig(
                service_type=ServiceType.CAMERA_VIEW,
                enabled=True,
                device="cpu",
                params=CameraViewServiceParams(),
            ),
        ],
        inference_settings=InferenceSettings(
            target_fps=25,
            processing_resolution=[1920, 1080],
            frame_skip=1,
        ),
    )
