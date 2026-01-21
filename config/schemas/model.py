"""
Model Configuration (Tier 1)

Simple model definitions - just what's needed to load and run a model.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from config.schemas.enums import ModelType, ModelArchitecture


class ModelParams(BaseModel):
    """Default inference parameters for a model."""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    batch_size: int = Field(default=1, ge=1)
    half_precision: bool = Field(default=False)


class ModelConfig(BaseModel):
    """
    Model configuration.

    Defines a model with its source and default parameters.
    Class mapping is handled automatically by the class registry.
    """
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(default="", description="Human-readable name")
    model_type: ModelType = Field(default=ModelType.OBJECT_DETECTION)
    model_architecture: ModelArchitecture = Field(default=ModelArchitecture.YOLO_V8)
    model_url: str = Field(default="", description="URL or path to model file")

    # Classes to detect (user-friendly names like "ball", "person")
    # Empty list = detect all classes the model supports
    classes_to_detect: List[str] = Field(
        default_factory=list,
        description="Class names to detect (e.g., ['ball', 'person'])"
    )

    # Default parameters
    default_params: ModelParams = Field(default_factory=ModelParams)

    # GPU required hint
    requires_gpu: bool = Field(default=True)

    class Config:
        protected_namespaces = ()


class ModelRegistryConfig(BaseModel):
    """Collection of available models."""
    models: List[ModelConfig] = Field(default_factory=list)

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get model config by ID."""
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None

    def list_model_ids(self) -> List[str]:
        """Get all model IDs."""
        return [m.model_id for m in self.models]
