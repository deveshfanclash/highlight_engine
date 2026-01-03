"""
Configuration Module

Provides configuration schemas and utilities for the inference system.
"""

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
