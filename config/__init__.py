"""
Configuration Module

Provides configuration schemas and utilities for the inference system.

Two schema versions available:
- v1 (schemas.py): Original schemas with enums (legacy)
- v2 (schemas_v2.py): Simplified schemas with string types (recommended)

Usage (v2 - recommended):
    from config import load_config, validate_config
    from config.schemas_v2 import GameConfig, ModelConfig, ServiceConfig

    config = load_config("path/to/config.yaml")
    validate_config(config)
"""

# =============================================================================
# V2 SCHEMAS (RECOMMENDED)
# =============================================================================

from config.schemas_v2 import (
    # Enums
    GameCategory as GameCategoryV2,
    MatchStatus as MatchStatusV2,
    # Config classes
    InputConfig,
    ClassMapping as ClassMappingV2,
    ModelConfig as ModelConfigV2,
    ServiceConfig as ServiceConfigV2,
    OutputConfig,
    GameConfig as GameConfigV2,
    MatchConfig as MatchConfigV2,
)

from config.loader_v2 import (
    load_config,
    load_config_from_mongo,
    load_config_auto,
    ConfigStore,
)

from config.validation import (
    validate_config,
    ConfigValidationError,
    ValidationResult,
    quick_validate,
)

# =============================================================================
# V1 SCHEMAS (LEGACY)
# =============================================================================

from config.schemas import (
    # Enums
    ModelType,
    ModelArchitecture,
    DeviceType,
    ServiceType,
    OutputFormat,
    GameCategory,
    MatchStatus,

    # Model Config
    ModelParams,
    CustomModelConfig,
    ClassMapping,
    ModelConfig,

    # Service Config
    ServiceParams,
    ODServiceParams,
    CameraViewServiceParams,
    ReplayDetectionServiceParams,
    ServiceConfig,

    # Main Configs
    InferenceSettings,
    GameConfig,
    MatchConfig,

    # Output Schemas
    BoundingBox,
    Detection,
    InferenceOutput,
    CameraViewOutput,

    # Factory Functions
    create_default_football_config,
    create_default_cricket_config,
)

from config.class_registry import (
    UniversalClassID,
    ClassDefinition,
    UNIVERSAL_CLASS_REGISTRY,
    get_class_by_name,
    get_class_by_id,
    get_classes_for_sport,
    get_class_id,
    get_class_name,
    validate_class_name,
    create_model_to_universal_mapping,
)

__all__ = [
    # ===================
    # V2 (Recommended)
    # ===================
    # Loaders
    "load_config",
    "load_config_from_mongo",
    "load_config_auto",
    "ConfigStore",
    # Validation
    "validate_config",
    "ConfigValidationError",
    "ValidationResult",
    "quick_validate",
    # V2 Schemas (aliased)
    "InputConfig",
    "OutputConfig",

    # ===================
    # V1 (Legacy)
    # ===================
    # Enums
    "ModelType",
    "ModelArchitecture",
    "DeviceType",
    "ServiceType",
    "OutputFormat",
    "GameCategory",
    "MatchStatus",

    # Model Config
    "ModelParams",
    "CustomModelConfig",
    "ClassMapping",
    "ModelConfig",

    # Service Config
    "ServiceParams",
    "ODServiceParams",
    "CameraViewServiceParams",
    "ReplayDetectionServiceParams",
    "ServiceConfig",

    # Main Configs
    "InferenceSettings",
    "GameConfig",
    "MatchConfig",

    # Output Schemas
    "BoundingBox",
    "Detection",
    "InferenceOutput",
    "CameraViewOutput",

    # Factory Functions
    "create_default_football_config",
    "create_default_cricket_config",

    # Class Registry
    "UniversalClassID",
    "ClassDefinition",
    "UNIVERSAL_CLASS_REGISTRY",
    "get_class_by_name",
    "get_class_by_id",
    "get_classes_for_sport",
    "get_class_id",
    "get_class_name",
    "validate_class_name",
    "create_model_to_universal_mapping",
]
