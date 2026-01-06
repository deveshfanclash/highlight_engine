"""
Configuration Schemas

Clean, typed configuration for the inference system.

Structure:
- enums.py: All enumeration types
- model.py: Model configurations (independent, supports A/B testing)
- service.py: Service configurations (typed per service)
- game.py: Game configurations (references models by ID)
- output.py: Output schemas for database writes
"""

# Enums
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

# Model configs
from config.schemas.model import (
    ClassMapping,
    ModelParams,
    ModelConfig,
    ModelRegistryConfig,
)

# Service configs
from config.schemas.service import (
    BaseServiceConfig,
    ODServiceConfig,
    CameraViewServiceConfig,
    SegmentationServiceConfig,
    ReplayDetectionServiceConfig,
    EventDetectionServiceConfig,
    AudioAnalysisServiceConfig,
    HLSMetadataServiceConfig,
    create_service_config,
)

# Game configs
from config.schemas.game import (
    InferenceSettings,
    GameConfig,
    MatchConfig,
    ServiceConfigUnion,
    create_service_from_dict,
)

# Output schemas
from config.schemas.output import (
    BoundingBox,
    Detection,
    InferenceOutput,
    CameraViewOutput,
    HLSMetadataOutput,
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
    # Model
    "ClassMapping",
    "ModelParams",
    "ModelConfig",
    "ModelRegistryConfig",
    # Service
    "BaseServiceConfig",
    "ODServiceConfig",
    "CameraViewServiceConfig",
    "SegmentationServiceConfig",
    "ReplayDetectionServiceConfig",
    "EventDetectionServiceConfig",
    "AudioAnalysisServiceConfig",
    "HLSMetadataServiceConfig",
    "create_service_config",
    # Game
    "InferenceSettings",
    "GameConfig",
    "MatchConfig",
    "ServiceConfigUnion",
    "create_service_from_dict",
    # Output
    "BoundingBox",
    "Detection",
    "InferenceOutput",
    "CameraViewOutput",
    "HLSMetadataOutput",
]
