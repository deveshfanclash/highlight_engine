"""
Configuration Module

4-Tier Configuration System:
- Tier 1: Model Registry (model.py) - Pure ML model definitions
- Tier 2: Game Template (game.py) - Sport-specific logic
- Tier 3: Deployment Profile (deployment.py) - Infrastructure configuration
- Tier 4: Match Config (match.py) - Runtime match configuration

Structure:
- config/schemas/ - Modular schema definitions
- config/loader.py - Load configs from YAML or MongoDB
- config/class_registry.py - Universal class definitions
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
    Environment,

    # Tier 1: Model
    ClassMapping,
    ModelParams,
    ResourceRequirements,
    ModelConfig,
    ModelRegistryConfig,

    # Tier 2: Game Template
    ModelAssignment,
    BaseServiceTemplate,
    ODServiceTemplate,
    CameraViewServiceTemplate,
    SegmentationServiceTemplate,
    HLSMetadataServiceTemplate,
    ReplayDetectionServiceTemplate,
    EventDetectionServiceTemplate,
    AudioAnalysisServiceTemplate,
    ServiceTemplateUnion,
    InferenceSettings,
    GameTemplate,
    create_service_template_from_dict,
    create_game_template,

    # Tier 3: Deployment
    DeploymentProfile,
    create_development_profile,
    create_production_profile,

    # Tier 4: Match
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
    CameraViewOutput,
    HLSMetadataOutput,
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
    "ServiceType",
    "InputType",
    "ProcessingPattern",
    "DeviceType",
    "GameCategory",
    "MatchStatus",
    "OutputFormat",
    "Environment",

    # Tier 1: Model
    "ClassMapping",
    "ModelParams",
    "ResourceRequirements",
    "ModelConfig",
    "ModelRegistryConfig",

    # Tier 2: Game Template
    "ModelAssignment",
    "BaseServiceTemplate",
    "ODServiceTemplate",
    "CameraViewServiceTemplate",
    "SegmentationServiceTemplate",
    "HLSMetadataServiceTemplate",
    "ReplayDetectionServiceTemplate",
    "EventDetectionServiceTemplate",
    "AudioAnalysisServiceTemplate",
    "ServiceTemplateUnion",
    "InferenceSettings",
    "GameTemplate",
    "create_service_template_from_dict",
    "create_game_template",

    # Tier 3: Deployment
    "DeploymentProfile",
    "create_development_profile",
    "create_production_profile",

    # Tier 4: Match
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
    "CameraViewOutput",
    "HLSMetadataOutput",

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
