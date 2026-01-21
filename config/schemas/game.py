"""
Game Template Configuration (Tier 2)

Defines WHAT to do for a sport - the inference logic.

Think of this as "the playbook" for a sport:
- Which models to use
- Which services to run
- How to interpret model outputs (class mapping)
- Default inference settings
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

from config.schemas.enums import (
    GameCategory,
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
    - What role it plays (primary OD, pose, etc.)
    - Which classes to detect (filtering)
    - Game-specific parameter overrides
    """
    model_id: str = Field(..., description="Reference to model in registry")
    role: str = Field(
        default="default",
        description="Role in game (e.g., 'default', 'pose')"
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
# SERVICE TEMPLATES
# =============================================================================

class BaseServiceTemplate(BaseModel):
    """Base template for service configuration."""
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


class PoseServiceTemplate(BaseServiceTemplate):
    """Pose Estimation service template"""
    service_type: ServiceType = Field(default=ServiceType.POSE_ESTIMATION)

    # Which model roles to run
    model_roles: List[str] = Field(
        default_factory=lambda: ["default"],
        description="Model roles to use from model_assignments"
    )

    # Processing settings
    target_width: Optional[int] = Field(None, description="Resize width")
    target_height: Optional[int] = Field(None, description="Resize height")

    # Keypoint settings
    keypoint_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence for keypoint visibility"
    )

    class Config:
        protected_namespaces = ()


# Union of all service templates
ServiceTemplateUnion = Union[
    ODServiceTemplate,
    PoseServiceTemplate,
    BaseServiceTemplate,
]


# =============================================================================
# INFERENCE SETTINGS
# =============================================================================

class InferenceSettings(BaseModel):
    """Global inference settings for a game."""
    target_fps: int = Field(default=25, description="Target FPS for processing")
    frame_skip: int = Field(
        default=1, ge=1,
        description="Global frame skip (can be overridden per service)"
    )
    processing_resolution: List[int] = Field(
        default=[1280, 720],
        description="Default processing resolution [width, height]"
    )


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
    """
    # Identity
    game_id: str = Field(..., description="Unique game identifier")
    game_name: str = Field(..., description="Human-readable name")
    description: str = Field(default="")

    # Model assignments
    model_assignments: List[ModelAssignment] = Field(
        default_factory=list,
        description="How models are used in this game"
    )

    # Service templates
    services: List[ServiceTemplateUnion] = Field(
        default_factory=list,
        description="Services to run for this game"
    )

    # Global settings
    inference_settings: InferenceSettings = Field(default_factory=InferenceSettings)

    # Universal class mapping (game-level interpretation)
    universal_class_definitions: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Game-specific class definitions"
    )

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

    def get_model_assignment(self, role: str) -> Optional[ModelAssignment]:
        """Get model assignment by role"""
        for assignment in self.model_assignments:
            if assignment.role == role:
                return assignment
        return None

    def get_model_ids(self) -> List[str]:
        """Get all model IDs referenced by this game"""
        return [a.model_id for a in self.model_assignments]


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
    model_ids: List[str],
    category: GameCategory = GameCategory.BALL_SPORT,
    enable_pose: bool = False,
) -> GameTemplate:
    """
    Factory function to create a basic game template.

    Creates a standard template with:
    - One OD service with specified models
    - Pose estimation service (optional)
    """
    assignments = [
        ModelAssignment(model_id=mid, role="default")
        for mid in model_ids
    ]

    services = [
        ODServiceTemplate(
            service_type=ServiceType.OBJECT_DETECTION,
            model_roles=["default"],
            enabled=True,
        ),
    ]

    if enable_pose:
        services.append(PoseServiceTemplate(
            service_type=ServiceType.POSE_ESTIMATION,
            model_roles=["pose"],
            enabled=True,
        ))

    return GameTemplate(
        game_id=game_id,
        game_name=game_name,
        model_assignments=assignments,
        services=services,
    )
