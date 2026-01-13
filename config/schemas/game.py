"""
Game Template Configuration (Tier 2)

Defines WHAT to do for a sport - the inference logic.
NO deployment/device information - that's in DeploymentProfile.

Think of this as "the playbook" for a sport:
- Which models to use
- Which services to run
- How to interpret model outputs (class mapping)
- Default inference settings
"""

from typing import List, Optional, Union, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from config.schemas.enums import (
    GameCategory,
    MatchStatus,
    InputType,
    ServiceType,
    ProcessingPattern,
)
from config.schemas.model import ModelParams, ClassMapping


# =============================================================================
# MODEL ASSIGNMENT
# =============================================================================

class ModelAssignment(BaseModel):
    """
    Defines how a model is used in a game.

    Specifies:
    - Which model to use (reference by ID)
    - What role it plays (primary OD, backup, segmentation, etc.)
    - Which classes to detect (filtering)
    - Game-specific parameter overrides
    """
    model_id: str = Field(..., description="Reference to model in registry")
    role: str = Field(
        default="default",
        description="Role in game (e.g., 'primary_od', 'ball_detector', 'segmentation')"
    )

    # Class filtering (which model classes to use)
    class_filter: List[int] = Field(
        default_factory=list,
        description="Model class IDs to use (empty = all)"
    )

    # Game-specific class mapping (overrides model defaults)
    class_mapping_override: List[ClassMapping] = Field(
        default_factory=list,
        description="Override class mapping for this game"
    )

    # Parameter overrides for this game
    params_override: Optional[ModelParams] = Field(
        None,
        description="Override model's default params for this game"
    )

    # Priority for this model (higher = processed first)
    priority: int = Field(default=0)

    class Config:
        protected_namespaces = ()


# =============================================================================
# SERVICE TEMPLATES (No device info)
# =============================================================================

class BaseServiceTemplate(BaseModel):
    """
    Base template for service configuration.

    NO device/instance information - that's in DeploymentProfile.
    This defines WHAT the service does, not WHERE it runs.
    """
    service_type: ServiceType = Field(..., description="Type of service")
    enabled: bool = Field(default=True, description="Whether service should run")

    # Processing settings
    processing_pattern: ProcessingPattern = Field(
        default=ProcessingPattern.FRAME_BY_FRAME
    )
    frame_skip: int = Field(default=1, ge=1, description="Process every Nth frame")

    # Output settings
    db_table_name: str = Field(default="inference_results")
    db_batch_size: int = Field(default=12, ge=1)
    db_flush_interval_ms: int = Field(default=250, ge=0)

    class Config:
        extra = "allow"


class ODServiceTemplate(BaseServiceTemplate):
    """Object Detection service template"""
    service_type: ServiceType = Field(default=ServiceType.OBJECT_DETECTION)

    # Which model roles to run
    model_roles: List[str] = Field(
        default_factory=lambda: ["default"],
        description="Model roles to use from model_assignments"
    )

    # Processing settings
    target_width: Optional[int] = Field(None, description="Resize width")
    target_height: Optional[int] = Field(None, description="Resize height")

    class Config:
        protected_namespaces = ()


class CameraViewServiceTemplate(BaseServiceTemplate):
    """Camera View Detection service template"""
    service_type: ServiceType = Field(default=ServiceType.CAMERA_VIEW)

    # Detection thresholds
    phash_threshold: int = Field(default=20)
    histogram_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    min_frame_gap: int = Field(default=0, ge=0)

    # Processing settings
    resolution_scale: float = Field(default=0.5, gt=0.0, le=1.0)


class SegmentationServiceTemplate(BaseServiceTemplate):
    """Segmentation service template"""
    service_type: ServiceType = Field(default=ServiceType.SEGMENTATION)

    model_roles: List[str] = Field(default_factory=lambda: ["segmentation"])
    target_width: Optional[int] = None
    target_height: Optional[int] = None

    # Output settings
    save_masks_to_s3: bool = Field(default=True)
    mask_format: str = Field(default="npz")

    class Config:
        protected_namespaces = ()


class HLSMetadataServiceTemplate(BaseServiceTemplate):
    """HLS Metadata service template"""
    service_type: ServiceType = Field(default=ServiceType.HLS_METADATA)

    poll_interval_seconds: int = Field(default=5)
    timeout_no_segment_seconds: int = Field(default=60)
    resolution_preference: str = Field(default="_480p.m3u8")


class ReplayDetectionServiceTemplate(BaseServiceTemplate):
    """Replay Detection service template"""
    service_type: ServiceType = Field(default=ServiceType.REPLAY_DETECTION)

    model_roles: List[str] = Field(default_factory=list)
    overlay_detection_threshold: float = Field(default=0.8)

    class Config:
        protected_namespaces = ()


class EventDetectionServiceTemplate(BaseServiceTemplate):
    """Event Detection service template"""
    service_type: ServiceType = Field(default=ServiceType.EVENT_DETECTION)
    processing_pattern: ProcessingPattern = Field(default=ProcessingPattern.CLIP_BASED)

    model_roles: List[str] = Field(default_factory=list)
    clip_duration_seconds: float = Field(default=10.0)
    clip_overlap_seconds: float = Field(default=2.0)

    class Config:
        protected_namespaces = ()


class AudioAnalysisServiceTemplate(BaseServiceTemplate):
    """Audio Analysis service template"""
    service_type: ServiceType = Field(default=ServiceType.AUDIO_ANALYSIS)
    processing_pattern: ProcessingPattern = Field(default=ProcessingPattern.AUDIO_BASED)

    model_roles: List[str] = Field(default_factory=list)
    sample_rate: int = Field(default=16000)
    chunk_duration_seconds: float = Field(default=30.0)

    class Config:
        protected_namespaces = ()


# Union of all service templates
ServiceTemplateUnion = Union[
    ODServiceTemplate,
    CameraViewServiceTemplate,
    SegmentationServiceTemplate,
    HLSMetadataServiceTemplate,
    ReplayDetectionServiceTemplate,
    EventDetectionServiceTemplate,
    AudioAnalysisServiceTemplate,
    BaseServiceTemplate,
]


# =============================================================================
# INFERENCE SETTINGS
# =============================================================================

class InferenceSettings(BaseModel):
    """
    Global inference settings for a game.

    These are defaults that can be overridden per-service or per-match.
    """
    target_fps: int = Field(default=25, description="Target FPS for processing")
    frame_skip: int = Field(
        default=1, ge=1,
        description="Global frame skip (can be overridden per service)"
    )
    processing_resolution: List[int] = Field(
        default=[1280, 720],
        description="Default processing resolution [width, height]"
    )
    # stream_buffer_size: int = Field(default=30)

    # Quality settings
    # enable_frame_interpolation: bool = Field(default=False)
    # max_frame_lag_ms: int = Field(
    #     default=1000,
    #     description="Maximum acceptable lag before dropping frames"
    # )


# =============================================================================
# GAME TEMPLATE
# =============================================================================

class GameTemplate(BaseModel):
    """
    Game Template Configuration (Tier 2).

    Defines the inference logic for a sport:
    - Which models to use (references to Tier 1)
    - How to use them (roles, class filtering)
    - Which services to run
    - Sport-specific class interpretation

    NO deployment/device information - that's in DeploymentProfile (Tier 3).
    """

    # -------------------------------------------------------------------------
    # IDENTITY
    # -------------------------------------------------------------------------
    game_id: str = Field(..., description="Unique game identifier")
    game_name: str = Field(..., description="Human-readable name (e.g., 'Football')")
    # game_category: GameCategory = Field(default=GameCategory.BALL_SPORT)
    description: str = Field(default="")

    # -------------------------------------------------------------------------
    # MODEL ASSIGNMENTS
    # -------------------------------------------------------------------------
    model_assignments: List[ModelAssignment] = Field(
        default_factory=list,
        description="How models are used in this game"
    )

    # -------------------------------------------------------------------------
    # SERVICE TEMPLATES
    # -------------------------------------------------------------------------
    services: List[ServiceTemplateUnion] = Field(
        default_factory=list,
        description="Services to run for this game"
    )

    # -------------------------------------------------------------------------
    # GLOBAL SETTINGS
    # -------------------------------------------------------------------------
    inference_settings: InferenceSettings = Field(default_factory=InferenceSettings)

    # -------------------------------------------------------------------------
    # UNIVERSAL CLASS MAPPING (Game-level interpretation)
    # -------------------------------------------------------------------------
    # This defines how this game interprets universal classes
    # For example, in football, PERSON might be split into PLAYER, REFEREE
    universal_class_definitions: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Game-specific class definitions and hierarchies"
    )

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------
    # config_version: int = Field(default=1)
    # created_at: Optional[datetime] = None
    # updated_at: Optional[datetime] = None
    # tags: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True
        protected_namespaces = ()

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def get_enabled_services(self) -> List[ServiceTemplateUnion]:
        """Get only enabled services"""
        return [s for s in self.services if s.enabled]

    def get_services_by_type(self, service_type: str) -> List[ServiceTemplateUnion]:
        """Get services of a specific type"""
        return [s for s in self.services if s.service_type.value == service_type]

    def get_model_assignment(self, role: str) -> Optional[ModelAssignment]:
        """Get model assignment by role"""
        for assignment in self.model_assignments:
            if assignment.role == role:
                return assignment
        return None

    def get_model_ids(self) -> List[str]:
        """Get all model IDs referenced by this game"""
        return [a.model_id for a in self.model_assignments]

    def get_model_ids_for_role(self, role: str) -> List[str]:
        """Get model IDs for a specific role"""
        return [a.model_id for a in self.model_assignments if a.role == role]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_service_template_from_dict(data: Dict[str, Any]) -> ServiceTemplateUnion:
    """
    Create typed service template from dict.

    Automatically selects correct template class based on service_type.
    """
    service_type = data.get("service_type", "")

    type_to_class = {
        "object_detection": ODServiceTemplate,
        "camera_view": CameraViewServiceTemplate,
        "segmentation": SegmentationServiceTemplate,
        "replay_detection": ReplayDetectionServiceTemplate,
        "event_detection": EventDetectionServiceTemplate,
        "audio_analysis": AudioAnalysisServiceTemplate,
        "hls_metadata": HLSMetadataServiceTemplate,
    }

    template_class = type_to_class.get(service_type, BaseServiceTemplate)
    return template_class(**data)


def create_game_template(
    game_id: str,
    game_name: str,
    model_ids: List[str],
    category: GameCategory = GameCategory.BALL_SPORT,
    enable_camera_view: bool = True,
    enable_hls_metadata: bool = True,
) -> GameTemplate:
    """
    Factory function to create a basic game template.

    Creates a standard template with:
    - One OD service with specified models
    - Camera view service (optional)
    - HLS metadata service (optional)
    """
    # Create model assignments from model_ids
    assignments = [
        ModelAssignment(model_id=mid, role="default")
        for mid in model_ids
    ]

    # Create services
    services = [
        ODServiceTemplate(
            service_type=ServiceType.OBJECT_DETECTION,
            model_roles=["default"],
            enabled=True,
        ),
    ]

    if enable_camera_view:
        services.append(CameraViewServiceTemplate(
            service_type=ServiceType.CAMERA_VIEW,
            enabled=True,
        ))

    if enable_hls_metadata:
        services.append(HLSMetadataServiceTemplate(
            service_type=ServiceType.HLS_METADATA,
            enabled=True,
        ))

    return GameTemplate(
        game_id=game_id,
        game_name=game_name,
        game_category=category,
        model_assignments=assignments,
        services=services,
    )
