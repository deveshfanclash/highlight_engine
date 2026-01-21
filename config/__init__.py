"""
Configuration Module

3-Tier Configuration System:
- Tier 1: Model Registry (model.py) - Pure ML model definitions
- Tier 2: Game Template (game.py) - Sport-specific logic
- Tier 3: Match Config (match.py) - Runtime match configuration

Infrastructure configuration is handled separately in config/environment.py

Structure:
- config/schemas/ - Modular schema definitions
- config/loader.py - Load configs from YAML or MongoDB
- config/class_registry.py - Universal class definitions
- config/environment.py - Infrastructure/env config
"""

# Import from schemas module
from config.schemas import (
    # Enums
    ModelType,
    ModelArchitecture,
    ServiceType,
    InputType,
    ProcessingPattern,
    DeviceType,
    GameCategory,
    MatchStatus,
    OutputFormat,

    # Tier 1: Model
    ModelParams,
    ModelConfig,
    ModelRegistryConfig,

    # Tier 2: Game Template
    ModelAssignment,
    BaseServiceTemplate,
    ODServiceTemplate,
    PoseServiceTemplate,
    ServiceTemplateUnion,
    InferenceSettings,
    GameTemplate,
    create_service_template_from_dict,
    create_game_template,

    # Tier 3: Match
    ResumePosition,
    ServiceOverride,
    MatchOverrides,
    MatchMetadata,
    MatchConfig,
    create_match_config,

    # Output Schemas
    BoundingBox,
    Detection,
    InferenceOutput,
    Keypoint,
    PoseDetection,
    PoseOutput,
)

from config.class_registry import (
    CLASS_ALIASES,
    normalize_class_name,
    find_model_class_ids,
    create_class_mapping,
    get_canonical_names,
    add_alias,
    COCO_CLASSES,
)

from config.environment import (
    InfraConfig,
    get_infra_config,
    reset_config,
    PROJECT_ROOT,
)

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

    # Tier 2: Game Template
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

    # Output Schemas
    "BoundingBox",
    "Detection",
    "InferenceOutput",
    "Keypoint",
    "PoseDetection",
    "PoseOutput",

    # Class Registry
    "CLASS_ALIASES",
    "normalize_class_name",
    "find_model_class_ids",
    "create_class_mapping",
    "get_canonical_names",
    "add_alias",
    "COCO_CLASSES",

    # Infrastructure Config
    "InfraConfig",
    "get_infra_config",
    "reset_config",
    "PROJECT_ROOT",
]
