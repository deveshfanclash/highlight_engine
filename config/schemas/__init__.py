"""
Configuration Schemas

4-Tier Configuration System:

Tier 1: Model Registry (model.py)
    - Pure ML model definitions
    - No deployment/device info
    - Versioned for A/B testing

Tier 2: Game Template (game.py)
    - Sport-specific logic
    - Model assignments with roles
    - Service templates (no device info)
    - Class mappings

Tier 3: Deployment Profile (deployment.py)
    - Infrastructure configuration
    - Service → Instance mapping
    - Batch queues and scaling
    - Monitoring and cost controls

Tier 4: Match Config (match.py)
    - Runtime match configuration
    - References game + deployment
    - Override mechanism
    - Status tracking
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
    ClassMapping,
    ModelParams,
    ResourceRequirements,
    ModelConfig,
    ModelRegistryConfig,
)

# =============================================================================
# TIER 2: GAME TEMPLATE
# =============================================================================
from config.schemas.game import (
    # Model assignment
    ModelAssignment,
    # Service templates
    BaseServiceTemplate,
    ODServiceTemplate,
    CameraViewServiceTemplate,
    SegmentationServiceTemplate,
    HLSMetadataServiceTemplate,
    ReplayDetectionServiceTemplate,
    EventDetectionServiceTemplate,
    AudioAnalysisServiceTemplate,
    ServiceTemplateUnion,
    # Settings
    InferenceSettings,
    # Main config
    GameTemplate,
    # Factory
    create_service_template_from_dict,
    create_game_template,
)

# =============================================================================
# TIER 3: DEPLOYMENT PROFILE
# =============================================================================
from config.schemas.deployment import (
    # Enums
    Environment,
    InstanceType,
    # Instance assignment
    InstanceAssignment,
    # Batch config
    BatchQueueConfig,
    BatchConfig,
    # Operational configs
    ScalingConfig,
    MonitoringConfig,
    CostControlConfig,
    DatabaseConfig,
    NetworkConfig,
    # Main config
    DeploymentProfile,
    # Factory functions
    create_development_profile,
    create_production_profile,
    create_multi_gpu_profile,
)

# =============================================================================
# TIER 4: MATCH CONFIG
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
    CameraViewOutput,
    HLSMetadataOutput,
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
    "Environment",
    "InstanceType",

    # Tier 1: Model
    "ClassMapping",
    "ModelParams",
    "ResourceRequirements",
    "ModelConfig",
    "ModelRegistryConfig",

    # Tier 2: Game
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
    "InstanceAssignment",
    "BatchQueueConfig",
    "BatchConfig",
    "ScalingConfig",
    "MonitoringConfig",
    "CostControlConfig",
    "DatabaseConfig",
    "NetworkConfig",
    "DeploymentProfile",
    "create_development_profile",
    "create_production_profile",
    "create_multi_gpu_profile",

    # Tier 4: Match
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
    "CameraViewOutput",
    "HLSMetadataOutput",

]
