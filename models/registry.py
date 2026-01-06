"""
Model Registry

Registry pattern for loading models by architecture string.
Allows adding new model types without code changes.

Usage:
    from models.registry import ModelRegistry, create_model_from_config

    # Register a custom loader
    ModelRegistry.register("my_model", MyModelLoader)

    # Create model from config
    model = create_model_from_config(model_config, device="cuda:0")
"""

import os
import logging
import importlib
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Any, List
from pathlib import Path

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# ABSTRACT MODEL LOADER
# =============================================================================

class ModelLoader(ABC):
    """
    Abstract base class for model loaders.

    Implement this to add support for new model architectures.
    """

    @classmethod
    @abstractmethod
    def load(
        cls,
        model_id: str,
        weights: str,
        device: str = "cpu",
        class_mapping: Optional[Dict[int, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> BaseModel:
        """
        Load a model.

        Args:
            model_id: Unique model identifier
            weights: Path to weights (local path, S3 URL, or model ID)
            device: Device to run on (cpu, cuda:0, etc.)
            class_mapping: Map model class IDs to universal class names
            params: Architecture-specific parameters

        Returns:
            Loaded model instance
        """
        pass

    @classmethod
    def resolve_weights(cls, weights: str, cache_dir: Optional[str] = None) -> str:
        """
        Resolve weights path - download from S3 if needed.

        Args:
            weights: S3 URL, local path, or model ID
            cache_dir: Directory for caching downloaded weights

        Returns:
            Local path to weights
        """
        if weights.startswith("s3://"):
            return cls._download_from_s3(weights, cache_dir)
        elif weights.startswith(("http://", "https://")):
            return cls._download_from_url(weights, cache_dir)
        else:
            # Local path or model ID (e.g., "yolov8n.pt")
            return weights

    @classmethod
    def _download_from_s3(cls, s3_url: str, cache_dir: Optional[str] = None) -> str:
        """Download weights from S3"""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 required for S3 downloads: pip install boto3")

        # Parse S3 URL
        # s3://bucket/path/to/model.pt -> bucket, path/to/model.pt
        parts = s3_url.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        # Determine local path
        cache_dir = cache_dir or os.path.expanduser("~/.cache/inference_models")
        local_path = Path(cache_dir) / bucket / key

        if local_path.exists():
            logger.info(f"Using cached model: {local_path}")
            return str(local_path)

        # Download
        logger.info(f"Downloading model from S3: {s3_url}")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        s3 = boto3.client("s3")
        s3.download_file(bucket, key, str(local_path))

        logger.info(f"Model downloaded to: {local_path}")
        return str(local_path)

    @classmethod
    def _download_from_url(cls, url: str, cache_dir: Optional[str] = None) -> str:
        """Download weights from HTTP(S) URL"""
        import urllib.request
        import hashlib

        cache_dir = cache_dir or os.path.expanduser("~/.cache/inference_models")

        # Create filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = url.split("/")[-1] or f"model_{url_hash}"
        local_path = Path(cache_dir) / "http" / filename

        if local_path.exists():
            logger.info(f"Using cached model: {local_path}")
            return str(local_path)

        logger.info(f"Downloading model from: {url}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(local_path))

        return str(local_path)


# =============================================================================
# BUILT-IN LOADERS
# =============================================================================

class YOLOLoader(ModelLoader):
    """Loader for Ultralytics YOLO models (v8, v11, v12)"""

    @classmethod
    def load(
        cls,
        model_id: str,
        weights: str,
        device: str = "cpu",
        class_mapping: Optional[Dict[int, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> BaseModel:
        from models.yolo_model import YOLOModel

        params = params or {}
        weights_path = cls.resolve_weights(weights)

        model = YOLOModel(
            model_id=model_id,
            device=device,
            class_mapping=class_mapping,
            half_precision=params.get("half_precision", False)
        )

        if not model.load(weights_path):
            raise RuntimeError(f"Failed to load YOLO model: {weights}")

        # Warmup if requested
        if params.get("warmup", True):
            model.warmup()

        return model


class WhisperLoader(ModelLoader):
    """Loader for OpenAI Whisper audio models"""

    @classmethod
    def load(
        cls,
        model_id: str,
        weights: str,
        device: str = "cpu",
        class_mapping: Optional[Dict[int, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> "WhisperModel":
        # Lazy import whisper model
        from models.whisper_model import WhisperModel

        params = params or {}
        model = WhisperModel(
            model_id=model_id,
            model_name=weights,  # e.g., "openai/whisper-base"
            device=device,
            language=params.get("language", "en"),
            task=params.get("task", "transcribe")
        )
        model.load()
        return model


class RTDETRLoader(ModelLoader):
    """Loader for RT-DETR models"""

    @classmethod
    def load(
        cls,
        model_id: str,
        weights: str,
        device: str = "cpu",
        class_mapping: Optional[Dict[int, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> BaseModel:
        # RT-DETR uses Ultralytics as well
        from models.yolo_model import YOLOModel

        params = params or {}
        weights_path = cls.resolve_weights(weights)

        model = YOLOModel(
            model_id=model_id,
            device=device,
            class_mapping=class_mapping,
            half_precision=params.get("half_precision", False)
        )

        if not model.load(weights_path):
            raise RuntimeError(f"Failed to load RT-DETR model: {weights}")

        return model


class CustomLoader(ModelLoader):
    """
    Loader for custom models specified via module/class path.

    Requires custom_loader config with:
    - module: Python module path (e.g., "my_models.custom")
    - class_name: Class name (e.g., "MyCustomModel")
    - load_method: Optional method name to call for loading
    """

    @classmethod
    def load(
        cls,
        model_id: str,
        weights: str,
        device: str = "cpu",
        class_mapping: Optional[Dict[int, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        custom_loader: Optional[Dict[str, str]] = None
    ) -> BaseModel:
        if not custom_loader:
            raise ValueError("custom_loader config required for architecture='custom'")

        module_path = custom_loader.get("module")
        class_name = custom_loader.get("class_name")
        load_method = custom_loader.get("load_method", "load")

        if not module_path or not class_name:
            raise ValueError("custom_loader must specify 'module' and 'class_name'")

        # Import the module and class
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {class_name} from {module_path}: {e}")

        # Create model instance
        model = model_class(
            model_id=model_id,
            device=device,
            class_mapping=class_mapping,
            **(params or {})
        )

        # Load weights
        weights_path = cls.resolve_weights(weights)
        load_fn = getattr(model, load_method)
        load_fn(weights_path)

        return model


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    """
    Registry for model architectures.

    Maps architecture strings to loader classes.
    Add new architectures by calling register().
    """

    _loaders: Dict[str, Type[ModelLoader]] = {}

    @classmethod
    def register(cls, architecture: str, loader: Type[ModelLoader]) -> None:
        """
        Register a model loader for an architecture.

        Args:
            architecture: Architecture name (e.g., "yolov8")
            loader: ModelLoader subclass
        """
        cls._loaders[architecture.lower()] = loader
        logger.debug(f"Registered model loader: {architecture}")

    @classmethod
    def get_loader(cls, architecture: str) -> Type[ModelLoader]:
        """
        Get loader for architecture.

        Args:
            architecture: Architecture name

        Returns:
            ModelLoader class

        Raises:
            ValueError if architecture not registered
        """
        key = architecture.lower()
        if key not in cls._loaders:
            available = list(cls._loaders.keys())
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Available: {available}. "
                f"Register new architectures with ModelRegistry.register()"
            )
        return cls._loaders[key]

    @classmethod
    def list_architectures(cls) -> List[str]:
        """List all registered architectures"""
        return list(cls._loaders.keys())

    @classmethod
    def is_registered(cls, architecture: str) -> bool:
        """Check if architecture is registered"""
        return architecture.lower() in cls._loaders


# =============================================================================
# REGISTER BUILT-IN LOADERS
# =============================================================================

# YOLO family
ModelRegistry.register("yolov8", YOLOLoader)
ModelRegistry.register("yolov11", YOLOLoader)
ModelRegistry.register("yolov12", YOLOLoader)
ModelRegistry.register("yolo", YOLOLoader)

# RT-DETR
ModelRegistry.register("rtdetr", RTDETRLoader)
ModelRegistry.register("rf_detr", RTDETRLoader)

# Audio
ModelRegistry.register("whisper", WhisperLoader)

# Custom
ModelRegistry.register("custom", CustomLoader)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_model(
    model_id: str,
    architecture: str,
    weights: str,
    device: str = "cpu",
    class_mapping: Optional[Dict[int, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    custom_loader: Optional[Dict[str, str]] = None
) -> BaseModel:
    """
    Create and load a model by architecture.

    Args:
        model_id: Unique model identifier
        architecture: Model architecture (yolov8, whisper, etc.)
        weights: Path to weights
        device: Device to run on
        class_mapping: Class ID to name mapping
        params: Architecture-specific parameters
        custom_loader: Custom loader config (for architecture="custom")

    Returns:
        Loaded model instance
    """
    loader_class = ModelRegistry.get_loader(architecture)

    # Handle custom loader specially
    if architecture.lower() == "custom":
        return CustomLoader.load(
            model_id=model_id,
            weights=weights,
            device=device,
            class_mapping=class_mapping,
            params=params,
            custom_loader=custom_loader
        )

    return loader_class.load(
        model_id=model_id,
        weights=weights,
        device=device,
        class_mapping=class_mapping,
        params=params
    )


def create_model_from_config(
    model_config: "ModelConfig",
    device: str = "cpu"
) -> BaseModel:
    """
    Create model from ModelConfig (v2 schema).

    Args:
        model_config: ModelConfig from config/schemas_v2.py
        device: Device to run on (overrides config)

    Returns:
        Loaded model instance
    """
    # Build class mapping from config
    class_mapping = {}
    for cls in model_config.classes:
        class_mapping[cls.model_class] = cls.name

    return create_model(
        model_id=model_config.id,
        architecture=model_config.architecture,
        weights=model_config.weights,
        device=device,
        class_mapping=class_mapping if class_mapping else None,
        params=model_config.params,
        custom_loader=model_config.custom_loader
    )


def load_game_models(
    game_config: "GameConfig",
    device_override: Optional[str] = None
) -> Dict[str, BaseModel]:
    """
    Load all models for a game config.

    Args:
        game_config: GameConfig from config/schemas_v2.py
        device_override: Override device for all models

    Returns:
        Dict mapping model_id to loaded model
    """
    models = {}

    for model_config in game_config.models:
        device = device_override or "cpu"

        try:
            logger.info(f"Loading model: {model_config.id} ({model_config.architecture})")
            models[model_config.id] = create_model_from_config(model_config, device)
            logger.info(f"Model loaded: {model_config.id}")
        except Exception as e:
            logger.error(f"Failed to load model {model_config.id}: {e}")
            raise

    return models


def load_service_models(
    game_config: "GameConfig",
    service_id: str,
    device: str = "cpu"
) -> Dict[str, BaseModel]:
    """
    Load only the models needed for a specific service.

    Args:
        game_config: GameConfig
        service_id: Service ID
        device: Device to run on

    Returns:
        Dict mapping model_id to loaded model
    """
    service = game_config.get_service(service_id)
    if not service:
        raise ValueError(f"Service not found: {service_id}")

    models = {}
    for model_id in service.models:
        model_config = game_config.get_model(model_id)
        if not model_config:
            raise ValueError(f"Model not found: {model_id}")

        logger.info(f"Loading model for service {service_id}: {model_id}")
        models[model_id] = create_model_from_config(model_config, device)

    return models


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "ModelLoader",
    "ModelRegistry",
    # Built-in loaders
    "YOLOLoader",
    "WhisperLoader",
    "RTDETRLoader",
    "CustomLoader",
    # Functions
    "create_model",
    "create_model_from_config",
    "load_game_models",
    "load_service_models",
]
