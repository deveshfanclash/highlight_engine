"""
Configuration Schemas

2-Tier Configuration System:

Tier 1: Model Registry (model.py)
    - Pure ML model definitions
    - No deployment/device info
    - Class mappings and inference params

Tier 2: Game Template (game.py)
    - Sport-specific logic
    - Service templates (each service references model_id directly)

Infrastructure config is handled separately in config/environment.py
"""

# =============================================================================
# ENUMS
# =============================================================================
from config.schemas.enums import (
    ModelType,
    ModelArchitecture,
    ServiceType,
    InputType,
    DeviceType,
    GameCategory,
    MatchStatus,
    OutputFormat,
)

# =============================================================================
# TIER 1: MODEL REGISTRY
# =============================================================================
from config.schemas.model import (
    ModelParams,
    ModelConfig,
    ModelRegistryConfig,
)

# =============================================================================
# TIER 2: GAME TEMPLATE
# =============================================================================
from config.schemas.game import (
    # Defaults
    InferenceDefaults,
    OutputDefaults,
    Defaults,
    # Overrides
    InferenceOverrides,
    OutputOverrides,
    # Service templates
    BaseServiceTemplate,
    ODServiceTemplate,
    PoseServiceTemplate,
    ServiceTemplateUnion,
    # Game template
    GameTemplate,
    create_service_template_from_dict,
)

# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================
from config.schemas.output import (
    BoundingBox,
    Detection,
    InferenceOutput,
    Keypoint,
    PoseDetection,
    PoseOutput,
)


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    # Enums
    "ModelType",
    "ModelArchitecture",
    "ServiceType",
    "InputType",
    "DeviceType",
    "GameCategory",
    "MatchStatus",
    "OutputFormat",

    # Tier 1: Model
    "ModelParams",
    "ModelConfig",
    "ModelRegistryConfig",

    # Tier 2: Game - Defaults
    "InferenceDefaults",
    "OutputDefaults",
    "Defaults",
    "InferenceOverrides",
    "OutputOverrides",
    # Tier 2: Game - Services
    "BaseServiceTemplate",
    "ODServiceTemplate",
    "PoseServiceTemplate",
    "ServiceTemplateUnion",
    "GameTemplate",
    "create_service_template_from_dict",

    # Output
    "BoundingBox",
    "Detection",
    "InferenceOutput",
    "Keypoint",
    "PoseDetection",
    "PoseOutput",
]
