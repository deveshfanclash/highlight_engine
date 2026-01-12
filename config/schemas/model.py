"""
Model Configuration (Tier 1)

Pure ML model definitions - NO deployment/device information.
Think of this as "the model file metadata".

Models are:
- Defined independently from games
- Referenced by model_id from GameTemplate
- Versioned for A/B testing
- Reusable across all games/sports
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from config.schemas.enums import ModelType, ModelArchitecture, OutputFormat


# =============================================================================
# CLASS MAPPING
# =============================================================================

class ClassMapping(BaseModel):
    """
    Maps model's native class ID to universal class name.

    Example:
        model outputs class_id=0 as "person"
        we map it to universal name "PLAYER" for football
    """
    model_class_id: int = Field(..., description="Class ID in model's output")
    model_class_name: str = Field(default="", description="Model's native class name")
    universal_class_name: str = Field(..., description="Universal class name (e.g., PERSON, BALL)")

    # Per-class threshold override (optional)
    confidence_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Override confidence threshold for this specific class"
    )

    class Config:
        protected_namespaces = ()


# =============================================================================
# MODEL PARAMETERS (Defaults)
# =============================================================================

class ModelParams(BaseModel):
    """
    Default inference parameters for a model.

    These can be overridden at:
    - Game level (GameTemplate.model_assignments[].params_override)
    - Match level (MatchConfig.overrides.model_params)
    """
    # Core inference params
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1)

    # Batching
    batch_size: int = Field(default=1, ge=1)

    # Input preprocessing
    input_size: Optional[List[int]] = Field(
        None,
        description="Model input size [width, height]. None = use model default."
    )

    # Precision
    half_precision: bool = Field(default=False, description="Use FP16 inference")

    # Model-specific extra params (for custom models)
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific parameters"
    )


# =============================================================================
# RESOURCE REQUIREMENTS (Hints, not assignments)
# =============================================================================

class ResourceRequirements(BaseModel):
    """
    Resource hints for the model.

    These are HINTS for the deployment system, not assignments.
    Actual instance selection is done by DeploymentProfile.
    """
    # GPU requirements
    requires_gpu: bool = Field(default=True, description="Does this model need a GPU?")
    min_gpu_memory_mb: int = Field(
        default=4000,
        description="Minimum GPU memory in MB"
    )
    recommended_gpu_memory_mb: int = Field(
        default=8000,
        description="Recommended GPU memory for optimal performance"
    )

    # CPU requirements
    min_cpu_cores: int = Field(default=2)
    min_memory_mb: int = Field(default=4000)

    # Performance hints
    estimated_inference_ms: int = Field(
        default=50,
        description="Expected inference time per frame (for capacity planning)"
    )
    supports_batching: bool = Field(
        default=True,
        description="Can this model process batches efficiently?"
    )
    max_batch_size: int = Field(
        default=32,
        description="Maximum efficient batch size"
    )


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

class ModelConfig(BaseModel):
    """
    Independent model configuration (Tier 1).

    Defines a single ML model with:
    - Identity and versioning
    - Source location
    - Native class definitions
    - Default inference parameters
    - Resource requirements (hints)

    NO deployment/device information - that's in DeploymentProfile.
    """

    # -------------------------------------------------------------------------
    # IDENTITY
    # -------------------------------------------------------------------------
    model_id: str = Field(
        ...,
        description="Unique model identifier (e.g., 'yolo_football_v2')"
    )
    model_name: str = Field(
        default="",
        description="Human-readable name"
    )
    description: str = Field(
        default="",
        description="Model description"
    )

    # -------------------------------------------------------------------------
    # TYPE AND ARCHITECTURE
    # -------------------------------------------------------------------------
    model_type: ModelType = Field(
        ...,
        description="Type of model (object_detection, segmentation, etc.)"
    )
    model_architecture: ModelArchitecture = Field(
        default=ModelArchitecture.YOLO_V8,
        description="Model architecture"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.BBOX,
        description="Output format (bbox, mask, keypoints, etc.)"
    )

    # -------------------------------------------------------------------------
    # SOURCE
    # -------------------------------------------------------------------------
    model_url: str = Field(
        ...,
        description="URL to download model weights (S3, HTTP, etc.)"
    )
    version: str = Field(
        default="latest",
        description="Model version for A/B testing and rollback"
    )
    checksum: Optional[str] = Field(
        None,
        description="SHA256 checksum for integrity verification"
    )

    # -------------------------------------------------------------------------
    # NATIVE CLASSES (What the model actually outputs)
    # -------------------------------------------------------------------------
    native_classes: List[str] = Field(
        default_factory=list,
        description="Class names the model outputs (e.g., ['person', 'ball', 'goal'])"
    )
    native_class_ids: List[int] = Field(
        default_factory=list,
        description="Class IDs corresponding to native_classes"
    )

    # -------------------------------------------------------------------------
    # CLASS MAPPING (Optional - can also be defined at game level)
    # -------------------------------------------------------------------------
    default_class_mapping: List[ClassMapping] = Field(
        default_factory=list,
        description="Default mapping from native classes to universal names"
    )

    # -------------------------------------------------------------------------
    # INFERENCE PARAMETERS
    # -------------------------------------------------------------------------
    default_params: ModelParams = Field(
        default_factory=ModelParams,
        description="Default inference parameters (can be overridden)"
    )

    # -------------------------------------------------------------------------
    # RESOURCE REQUIREMENTS
    # -------------------------------------------------------------------------
    resources: ResourceRequirements = Field(
        default_factory=ResourceRequirements,
        description="Resource hints for deployment"
    )

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for filtering (e.g., ['football', 'v2', 'production'])"
    )
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        protected_namespaces = ()

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def get_universal_class_name(self, model_class_id: int) -> Optional[str]:
        """Get universal class name for a model class ID"""
        for mapping in self.default_class_mapping:
            if mapping.model_class_id == model_class_id:
                return mapping.universal_class_name
        return None

    def get_class_mapping_dict(self) -> Dict[int, str]:
        """Get class mapping as dict {model_class_id: universal_name}"""
        return {m.model_class_id: m.universal_class_name for m in self.default_class_mapping}

    def get_native_class_name(self, class_id: int) -> Optional[str]:
        """Get native class name by ID"""
        if class_id < len(self.native_classes):
            return self.native_classes[class_id]
        return None


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class ModelRegistryConfig(BaseModel):
    """
    Model Registry - collection of all available models.

    Games reference models from this registry by model_id.
    Supports:
    - Listing models by type
    - Version filtering for A/B testing
    - Tag-based filtering
    """
    models: List[ModelConfig] = Field(default_factory=list)

    def get_model(
        self,
        model_id: str,
        version: str = "latest"
    ) -> Optional[ModelConfig]:
        """Get model config by ID and version"""
        for model in self.models:
            if model.model_id == model_id:
                if version == "latest" or model.version == version:
                    return model
        return None

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelConfig]:
        """List models with optional filters"""
        result = self.models

        if model_type is not None:
            result = [m for m in result if m.model_type == model_type]

        if tags:
            result = [m for m in result if any(t in m.tags for t in tags)]

        return result

    def list_model_ids(self) -> List[str]:
        """Get all model IDs"""
        return [m.model_id for m in self.models]

    def get_models_by_ids(self, model_ids: List[str]) -> List[ModelConfig]:
        """Get multiple models by their IDs"""
        return [m for m in self.models if m.model_id in model_ids]
