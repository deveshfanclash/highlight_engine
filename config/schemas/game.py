"""
Game Configuration

Defines what models and services a game uses.
Models are referenced by ID (defined separately in model registry).
"""

from typing import List, Optional, Union, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from config.schemas.enums import GameCategory, MatchStatus, InputType
from config.schemas.service import (
    BaseServiceConfig,
    ODServiceConfig,
    CameraViewServiceConfig,
    SegmentationServiceConfig,
    ReplayDetectionServiceConfig,
    EventDetectionServiceConfig,
    AudioAnalysisServiceConfig,
    HLSMetadataServiceConfig,
)

# Union of all service config types
ServiceConfigUnion = Union[
    ODServiceConfig,
    CameraViewServiceConfig,
    SegmentationServiceConfig,
    ReplayDetectionServiceConfig,
    EventDetectionServiceConfig,
    AudioAnalysisServiceConfig,
    HLSMetadataServiceConfig,
    BaseServiceConfig,
]


class InferenceSettings(BaseModel):
    """Global inference settings for a game"""
    target_fps: int = Field(default=25, description="Target FPS for processing")
    frame_skip: int = Field(default=1, ge=1, description="Global frame skip (can be overridden per service)")
    processing_resolution: List[int] = Field(
        default=[1280, 720],
        description="Default processing resolution [width, height]"
    )
    stream_buffer_size: int = Field(default=30)


class GameConfig(BaseModel):
    """
    Configuration for a game/sport.

    Defines:
    - Which models to use (references model_ids)
    - Which services to run
    - Global inference settings

    Models are NOT embedded - they're referenced by ID from ModelRegistry.
    """
    # Identification
    game_id: str = Field(..., description="Unique game identifier")
    game_name: str = Field(..., description="Human-readable name")
    game_category: GameCategory = Field(default=GameCategory.BALL_SPORT)

    # Model references (IDs only, actual configs in ModelRegistry)
    model_ids: List[str] = Field(
        default_factory=list,
        description="Model IDs to load for this game (from ModelRegistry)"
    )

    # Services to run
    services: List[ServiceConfigUnion] = Field(
        default_factory=list,
        description="Services to run for this game"
    )

    # Global settings
    inference_settings: InferenceSettings = Field(default_factory=InferenceSettings)

    # Metadata
    config_version: int = Field(default=1)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        # Needed for Union type discrimination
        use_enum_values = True
        protected_namespaces = ()

    def get_enabled_services(self) -> List[ServiceConfigUnion]:
        """Get only enabled services"""
        return [s for s in self.services if s.enabled]

    def get_services_by_type(self, service_type: str) -> List[ServiceConfigUnion]:
        """Get services of a specific type"""
        return [s for s in self.services if s.service_type.value == service_type]


class MatchConfig(BaseModel):
    """
    Runtime configuration for a specific match.

    Created when a match starts, contains:
    - Match identifiers
    - Stream information
    - Reference to game configuration
    """
    # Identification
    match_id: str = Field(..., description="Unique match identifier")
    game_id: str = Field(..., description="Reference to game config")

    # Stream configuration
    stream_url: str = Field(..., description="Input stream URL")
    stream_type: InputType = Field(default=InputType.HLS)

    # Status
    status: MatchStatus = Field(default=MatchStatus.PENDING)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Resume support
    last_processed_frame: int = Field(default=0)
    last_processed_segment: int = Field(default=1)

    # Metadata
    league: Optional[str] = None
    tournament_id: Optional[str] = None
    tournament_name: Optional[str] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_service_from_dict(data: Dict[str, Any]) -> ServiceConfigUnion:
    """
    Create typed service config from dict.

    Automatically selects correct config class based on service_type.
    """
    service_type = data.get("service_type", "")

    type_to_class = {
        "object_detection": ODServiceConfig,
        "camera_view": CameraViewServiceConfig,
        "segmentation": SegmentationServiceConfig,
        "replay_detection": ReplayDetectionServiceConfig,
        "event_detection": EventDetectionServiceConfig,
        "audio_analysis": AudioAnalysisServiceConfig,
        "hls_metadata": HLSMetadataServiceConfig,
    }

    config_class = type_to_class.get(service_type, BaseServiceConfig)
    return config_class(**data)
