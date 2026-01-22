"""
Game Template Configuration (Tier 2)

Defines WHAT to do for a sport - the inference logic.

Think of this as "the playbook" for a sport:
- Which models to use
- Which services to run
- How to interpret model outputs (class mapping)
- Default settings with per-service overrides (hybrid pattern)
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

from config.schemas.enums import (
    GameCategory,
    ServiceType,
    ProcessingPattern,
)


# =============================================================================
# DEFAULTS (Global settings that can be overridden per-service)
# =============================================================================

class InferenceDefaults(BaseModel):
    """Default inference settings - can be overridden per service."""
    frame_skip: int = Field(default=1, ge=1, description="Process every Nth frame")
    processing_resolution: List[int] = Field(
        default=[1280, 720],
        description="Default processing resolution [width, height]"
    )

    # Parallelism settings
    num_workers: int = Field(
        default=1, ge=1,
        description="Number of worker processes for parallel inference"
    )
    worker_queue_size: int = Field(
        default=0, ge=0,
        description="Queue size per worker (0 = auto: num_workers * 4)"
    )

    # Buffer settings
    enable_buffering: Optional[bool] = Field(
        default=None,
        description="Enable frame buffering (None = auto-detect based on source type)"
    )
    buffer_size: int = Field(
        default=30, ge=1,
        description="Buffer size for frame buffering"
    )
    buffer_mode: str = Field(
        default="drop_old",
        description="Buffer mode: 'fifo' or 'drop_old'"
    )


class OutputDefaults(BaseModel):
    """Default output settings - can be overridden per service."""
    db_table_name: str = Field(default="inference_results")
    db_batch_size: int = Field(default=12, ge=1)
    db_flush_interval_ms: int = Field(default=250, ge=0)


class Defaults(BaseModel):
    """Container for all default settings."""
    inference: InferenceDefaults = Field(default_factory=InferenceDefaults)
    output: OutputDefaults = Field(default_factory=OutputDefaults)


# =============================================================================
# SERVICE OVERRIDE SCHEMAS
# =============================================================================

class InferenceOverrides(BaseModel):
    """Per-service inference overrides. Only specify what differs from defaults."""
    frame_skip: Optional[int] = Field(None, ge=1)
    processing_resolution: Optional[List[int]] = None
    num_workers: Optional[int] = Field(None, ge=1)
    worker_queue_size: Optional[int] = Field(None, ge=0)
    enable_buffering: Optional[bool] = None
    buffer_size: Optional[int] = Field(None, ge=1)
    buffer_mode: Optional[str] = None


class OutputOverrides(BaseModel):
    """Per-service output overrides. Only specify what differs from defaults."""
    db_table_name: Optional[str] = None
    db_batch_size: Optional[int] = Field(None, ge=1)
    db_flush_interval_ms: Optional[int] = Field(None, ge=0)


# =============================================================================
# SERVICE TEMPLATES
# =============================================================================

class BaseServiceTemplate(BaseModel):
    """Base template for service configuration."""
    service_type: ServiceType = Field(..., description="Type of service")
    enabled: bool = Field(default=True, description="Whether service should run")

    # Device configuration (explicit per-service, no global default)
    device: str = Field(
        default="cuda:0",
        description="Device to run on: 'cpu', 'cuda:0', 'cuda:1', etc."
    )

    # Processing settings
    processing_pattern: ProcessingPattern = Field(
        default=ProcessingPattern.FRAME_BY_FRAME
    )

    # Per-service overrides (optional - only specify what differs from defaults)
    inference_overrides: Optional[InferenceOverrides] = None
    output_overrides: Optional[OutputOverrides] = None

    model_config = {"extra": "allow", "protected_namespaces": ()}


class ODServiceTemplate(BaseServiceTemplate):
    """Object Detection service template"""
    service_type: ServiceType = Field(default=ServiceType.OBJECT_DETECTION)

    # Model to use (direct reference to model_id in models section)
    model_id: str = Field(..., description="Model ID to use for this service")

    # Processing settings
    target_width: Optional[int] = Field(None, description="Resize width")
    target_height: Optional[int] = Field(None, description="Resize height")

    model_config = {"protected_namespaces": ()}


class PoseServiceTemplate(BaseServiceTemplate):
    """Pose Estimation service template"""
    service_type: ServiceType = Field(default=ServiceType.POSE_ESTIMATION)

    # Model to use (direct reference to model_id in models section)
    model_id: str = Field(..., description="Model ID to use for this service")

    # Processing settings
    target_width: Optional[int] = Field(None, description="Resize width")
    target_height: Optional[int] = Field(None, description="Resize height")

    # Keypoint settings
    keypoint_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence for keypoint visibility"
    )

    model_config = {"protected_namespaces": ()}


# Union of all service templates
ServiceTemplateUnion = Union[
    ODServiceTemplate,
    PoseServiceTemplate,
    BaseServiceTemplate,
]


# =============================================================================
# GAME TEMPLATE
# =============================================================================

class GameTemplate(BaseModel):
    """
    Game Template Configuration (Tier 2).

    Defines the inference logic for a sport:
    - Which models to use
    - Which services to run (each service references a model directly)
    - Default settings with per-service override support
    """
    # Identity
    game_id: str = Field(..., description="Unique game identifier")
    game_name: str = Field(..., description="Human-readable name")
    description: str = Field(default="")

    # Service templates (each service specifies its model_id directly)
    services: List[ServiceTemplateUnion] = Field(
        default_factory=list,
        description="Services to run for this game"
    )

    # Global defaults (can be overridden per-service)
    defaults: Defaults = Field(default_factory=Defaults)

    # Game category
    game_category: GameCategory = Field(default=GameCategory.BALL_SPORT)

    class Config:
        use_enum_values = True
        protected_namespaces = ()

    # Helper methods
    def get_enabled_services(self) -> List[ServiceTemplateUnion]:
        """Get only enabled services"""
        return [s for s in self.services if s.enabled]

    def get_services_by_type(self, service_type: str) -> List[ServiceTemplateUnion]:
        """Get services of a specific type"""
        return [s for s in self.services if s.service_type.value == service_type]

    def get_model_ids(self) -> List[str]:
        """Get all model IDs referenced by services"""
        return [s.model_id for s in self.services if hasattr(s, 'model_id')]

    def resolve_inference_settings(self, service: ServiceTemplateUnion) -> Dict[str, Any]:
        """
        Resolve inference settings for a service.
        Merges global defaults with service-specific overrides.

        Args:
            service: The service template to resolve settings for

        Returns:
            Dict with resolved inference settings
        """
        # Start with global defaults
        resolved = self.defaults.inference.model_dump()

        # Apply service overrides if present
        if service.inference_overrides:
            overrides = service.inference_overrides.model_dump(exclude_none=True)
            resolved.update(overrides)

        return resolved

    def resolve_output_settings(self, service: ServiceTemplateUnion) -> Dict[str, Any]:
        """
        Resolve output settings for a service.
        Merges global defaults with service-specific overrides.

        Args:
            service: The service template to resolve settings for

        Returns:
            Dict with resolved output settings
        """
        # Start with global defaults
        resolved = self.defaults.output.model_dump()

        # Apply service overrides if present
        if service.output_overrides:
            overrides = service.output_overrides.model_dump(exclude_none=True)
            resolved.update(overrides)

        return resolved


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_service_template_from_dict(data: Dict[str, Any]) -> ServiceTemplateUnion:
    """Create typed service template from dict."""
    service_type = data.get("service_type", "")

    type_to_class = {
        "object_detection": ODServiceTemplate,
        "pose_estimation": PoseServiceTemplate,
    }

    template_class = type_to_class.get(service_type, BaseServiceTemplate)
    return template_class(**data)


def create_game_template(
    game_id: str,
    game_name: str,
    od_model_id: str,
    pose_model_id: Optional[str] = None,
    category: GameCategory = GameCategory.BALL_SPORT,
    defaults: Optional[Defaults] = None,
) -> GameTemplate:
    """
    Factory function to create a basic game template.

    Creates a standard template with:
    - One OD service with specified model
    - Pose estimation service (optional)
    - Default settings (or provided defaults)
    """
    services = [
        ODServiceTemplate(
            service_type=ServiceType.OBJECT_DETECTION,
            model_id=od_model_id,
            enabled=True,
        ),
    ]

    if pose_model_id:
        services.append(PoseServiceTemplate(
            service_type=ServiceType.POSE_ESTIMATION,
            model_id=pose_model_id,
            enabled=True,
        ))

    return GameTemplate(
        game_id=game_id,
        game_name=game_name,
        services=services,
        defaults=defaults or Defaults(),
    )
