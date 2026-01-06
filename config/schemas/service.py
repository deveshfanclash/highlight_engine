"""
Service Configuration

Typed configuration for each service type.
Each service has its own config class with specific parameters.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from config.schemas.enums import ServiceType, InputType, ProcessingPattern


# =============================================================================
# BASE SERVICE CONFIG
# =============================================================================

class BaseServiceConfig(BaseModel):
    """
    Base configuration for all services.

    Each service type extends this with specific parameters.
    """
    service_type: ServiceType = Field(..., description="Type of service")
    service_id: Optional[str] = Field(None, description="Unique ID (auto-generated if not set)")
    enabled: bool = Field(default=True, description="Whether service should run")
    device: str = Field(default="cpu", description="Device (cpu, cuda:0, cuda:1, etc.)")

    # Input configuration
    input_type: InputType = Field(default=InputType.HLS, description="Type of input to process")
    processing_pattern: ProcessingPattern = Field(
        default=ProcessingPattern.FRAME_BY_FRAME,
        description="How service processes input"
    )

    # Output configuration
    db_table_name: str = Field(default="inference_results")
    db_batch_size: int = Field(default=12, ge=1)
    db_flush_interval_ms: int = Field(default=250, ge=0)

    class Config:
        extra = "allow"  # Allow additional fields for extensibility


# =============================================================================
# OBJECT DETECTION SERVICE CONFIG
# =============================================================================

class ODServiceConfig(BaseServiceConfig):
    """Configuration for Object Detection Service"""
    service_type: ServiceType = Field(default=ServiceType.OBJECT_DETECTION)
    processing_pattern: ProcessingPattern = Field(default=ProcessingPattern.FRAME_BY_FRAME)

    # Models to run (references model_ids from ModelRegistry)
    model_ids: List[str] = Field(..., description="List of model IDs to run")

    # Processing settings
    frame_skip: int = Field(default=1, ge=1, description="Process every Nth frame")
    target_width: Optional[int] = Field(None, description="Resize width (None = source)")
    target_height: Optional[int] = Field(None, description="Resize height (None = source)")

    class Config:
        protected_namespaces = ()


# =============================================================================
# CAMERA VIEW SERVICE CONFIG
# =============================================================================

class CameraViewServiceConfig(BaseServiceConfig):
    """Configuration for Camera View Detection Service"""
    service_type: ServiceType = Field(default=ServiceType.CAMERA_VIEW)
    processing_pattern: ProcessingPattern = Field(default=ProcessingPattern.FRAME_BY_FRAME)

    # Detection thresholds
    phash_threshold: int = Field(default=20, description="Perceptual hash difference threshold")
    histogram_threshold: float = Field(
        default=0.90, ge=0.0, le=1.0,
        description="Histogram correlation threshold (below = cut detected)"
    )
    min_frame_gap: int = Field(
        default=0, ge=0,
        description="Minimum frames between cuts (0 = auto from fps)"
    )

    # Processing settings
    resolution_scale: float = Field(
        default=0.5, gt=0.0, le=1.0,
        description="Scale factor for processing resolution"
    )
    frame_skip: int = Field(default=1, ge=1)


# =============================================================================
# SEGMENTATION SERVICE CONFIG
# =============================================================================

class SegmentationServiceConfig(BaseServiceConfig):
    """Configuration for Segmentation Service"""
    service_type: ServiceType = Field(default=ServiceType.SEGMENTATION)
    processing_pattern: ProcessingPattern = Field(default=ProcessingPattern.FRAME_BY_FRAME)

    model_ids: List[str] = Field(..., description="Segmentation model IDs")
    frame_skip: int = Field(default=1, ge=1)
    target_width: Optional[int] = None
    target_height: Optional[int] = None

    # Output settings
    save_masks_to_s3: bool = Field(default=True, description="Save masks to S3")
    mask_format: str = Field(default="npz", description="Mask format (npz, png)")

    class Config:
        protected_namespaces = ()


# =============================================================================
# REPLAY DETECTION SERVICE CONFIG
# =============================================================================

class ReplayDetectionServiceConfig(BaseServiceConfig):
    """Configuration for Replay Detection Service (future)"""
    service_type: ServiceType = Field(default=ServiceType.REPLAY_DETECTION)
    processing_pattern: ProcessingPattern = Field(default=ProcessingPattern.FRAME_BY_FRAME)

    model_ids: List[str] = Field(default_factory=list, description="Model IDs if ML-based")

    # Detection settings
    overlay_detection_threshold: float = Field(
        default=0.8,
        description="Threshold for overlay detection"
    )

    class Config:
        protected_namespaces = ()


# =============================================================================
# EVENT DETECTION SERVICE CONFIG
# =============================================================================

class EventDetectionServiceConfig(BaseServiceConfig):
    """Configuration for Event Detection Service (future)"""
    service_type: ServiceType = Field(default=ServiceType.EVENT_DETECTION)
    processing_pattern: ProcessingPattern = Field(default=ProcessingPattern.CLIP_BASED)
    input_type: InputType = Field(default=InputType.CLIP)

    model_ids: List[str] = Field(default_factory=list)

    # Clip settings
    clip_duration_seconds: float = Field(default=10.0, description="Clip duration for analysis")
    clip_overlap_seconds: float = Field(default=2.0, description="Overlap between clips")

    class Config:
        protected_namespaces = ()


# =============================================================================
# AUDIO ANALYSIS SERVICE CONFIG
# =============================================================================

class AudioAnalysisServiceConfig(BaseServiceConfig):
    """Configuration for Audio Analysis Service (future)"""
    service_type: ServiceType = Field(default=ServiceType.AUDIO_ANALYSIS)
    processing_pattern: ProcessingPattern = Field(default=ProcessingPattern.AUDIO_BASED)
    input_type: InputType = Field(default=InputType.AUDIO)

    model_ids: List[str] = Field(default_factory=list)

    # Audio settings
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    chunk_duration_seconds: float = Field(default=30.0, description="Audio chunk duration")

    class Config:
        protected_namespaces = ()


# =============================================================================
# HLS METADATA SERVICE CONFIG
# =============================================================================

class HLSMetadataServiceConfig(BaseServiceConfig):
    """Configuration for HLS Metadata Service"""
    service_type: ServiceType = Field(default=ServiceType.HLS_METADATA)
    processing_pattern: ProcessingPattern = Field(default=ProcessingPattern.FRAME_BY_FRAME)
    device: str = Field(default="cpu")

    # Polling settings
    poll_interval_seconds: int = Field(default=5, description="M3U8 poll interval")
    timeout_no_segment_seconds: int = Field(default=60, description="Stop if no new segment")
    resolution_preference: str = Field(default="_480p.m3u8", description="Prefer low res for metadata")


# =============================================================================
# SERVICE CONFIG FACTORY
# =============================================================================

def create_service_config(service_type: str, **kwargs) -> BaseServiceConfig:
    """
    Factory function to create typed service config.

    Args:
        service_type: Service type string
        **kwargs: Service-specific parameters

    Returns:
        Typed service config instance
    """
    config_map = {
        ServiceType.OBJECT_DETECTION.value: ODServiceConfig,
        ServiceType.CAMERA_VIEW.value: CameraViewServiceConfig,
        ServiceType.SEGMENTATION.value: SegmentationServiceConfig,
        ServiceType.REPLAY_DETECTION.value: ReplayDetectionServiceConfig,
        ServiceType.EVENT_DETECTION.value: EventDetectionServiceConfig,
        ServiceType.AUDIO_ANALYSIS.value: AudioAnalysisServiceConfig,
        ServiceType.HLS_METADATA.value: HLSMetadataServiceConfig,
    }

    config_class = config_map.get(service_type, BaseServiceConfig)
    return config_class(**kwargs)
