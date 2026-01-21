"""
Configuration Schemas

3-Tier Configuration System:

Tier 1: Model Registry (model.py)
    - Pure ML model definitions
    - No deployment/device info
    - Class mappings and inference params

Tier 2: Game Template (game.py)
    - Sport-specific logic
    - Model assignments with roles
    - Service templates with settings

Tier 3: Match Config (match.py)
    - Runtime match configuration
    - Stream URL and match ID
    - Override mechanism

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
    ProcessingPattern,
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
    ModelAssignment,
    BaseServiceTemplate,
    ODServiceTemplate,
    PoseServiceTemplate,
    ServiceTemplateUnion,
    InferenceSettings,
    GameTemplate,
    create_service_template_from_dict,
    create_game_template,
)

# =============================================================================
# TIER 3: MATCH CONFIG
# =============================================================================
from config.schemas.match import (
    ResumePosition,
    ServiceOverride,
    MatchOverrides,
    MatchMetadata,
    MatchConfig,
    create_match_config,
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
    "ProcessingPattern",
    "DeviceType",
    "GameCategory",
    "MatchStatus",
    "OutputFormat",

    # Tier 1: Model
    "ModelParams",
    "ModelConfig",
    "ModelRegistryConfig",

    # Tier 2: Game
    "ModelAssignment",
    "BaseServiceTemplate",
    "ODServiceTemplate",
    "PoseServiceTemplate",
    "ServiceTemplateUnion",
    "InferenceSettings",
    "GameTemplate",
    "create_service_template_from_dict",
    "create_game_template",

    # Tier 3: Match
    "ResumePosition",
    "ServiceOverride",
    "MatchOverrides",
    "MatchMetadata",
    "MatchConfig",
    "create_match_config",

    # Output
    "BoundingBox",
    "Detection",
    "InferenceOutput",
    "Keypoint",
    "PoseDetection",
    "PoseOutput",
]
