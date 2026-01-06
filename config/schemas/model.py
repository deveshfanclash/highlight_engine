"""
Model Configuration

Independent model definitions that can be referenced by games.
Supports versioning and A/B testing.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from config.schemas.enums import ModelType, ModelArchitecture, OutputFormat


class ClassMapping(BaseModel):
    """Maps model's native class ID to universal class name"""
    model_class_id: int = Field(..., description="Class ID in model's output")
    universal_class_name: str = Field(..., description="Universal class name (e.g., PERSON, BALL)")
    confidence_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Override confidence threshold for this class"
    )

    class Config:
        protected_namespaces = ()


class ModelParams(BaseModel):
    """Inference parameters for a model"""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1)
    batch_size: int = Field(default=1, ge=1)
    input_size: Optional[List[int]] = Field(
        None,
        description="Model input size [width, height]. None = use model default."
    )
    half_precision: bool = Field(default=False, description="Use FP16 inference")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Model-specific params")


class ModelConfig(BaseModel):
    """
    Independent model configuration.

    Models are defined separately from games and referenced by model_id.
    This enables:
    - Reusing models across games
    - A/B testing with different versions
    - Independent model versioning
    """
    # Identification
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(default="", description="Human-readable name")
    model_type: ModelType = Field(..., description="Type of model")
    model_architecture: ModelArchitecture = Field(default=ModelArchitecture.YOLO_V8)

    # Source
    model_url: str = Field(..., description="URL to download model (S3, HTTP)")
    version: str = Field(default="latest", description="Model version for A/B testing")

    # Class configuration
    classes_to_predict: List[int] = Field(
        default_factory=list,
        description="Model's native class IDs to predict (empty = all)"
    )
    class_mapping: List[ClassMapping] = Field(
        default_factory=list,
        description="Map model class IDs to universal names"
    )

    # Inference parameters
    params: ModelParams = Field(default_factory=ModelParams)

    # Output
    output_format: OutputFormat = Field(default=OutputFormat.BBOX)

    # Device preference
    preferred_device: str = Field(default="gpu", description="Preferred device (cpu, gpu)")

    class Config:
        # Allow model_* fields without warning
        protected_namespaces = ()

    def get_class_name(self, model_class_id: int) -> Optional[str]:
        """Get universal class name for a model class ID"""
        for mapping in self.class_mapping:
            if mapping.model_class_id == model_class_id:
                return mapping.universal_class_name
        return None

    def get_class_mapping_dict(self) -> Dict[int, str]:
        """Get class mapping as dict {model_id: universal_name}"""
        return {m.model_class_id: m.universal_class_name for m in self.class_mapping}


class ModelRegistryConfig(BaseModel):
    """
    Configuration for the model registry.

    Defines all available models. Games reference these by model_id.
    """
    models: List[ModelConfig] = Field(default_factory=list)

    def get_model(self, model_id: str, version: str = "latest") -> Optional[ModelConfig]:
        """Get model config by ID and version"""
        for model in self.models:
            if model.model_id == model_id:
                if version == "latest" or model.version == version:
                    return model
        return None

    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelConfig]:
        """List all models, optionally filtered by type"""
        if model_type is None:
            return self.models
        return [m for m in self.models if m.model_type == model_type]
