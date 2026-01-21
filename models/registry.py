"""
Model Registry

Independent model management.
Models are defined separately from games and loaded on demand.
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

from config.schemas import (
    ModelConfig,
    ModelArchitecture,
)
from models.yolo_model import YOLOModel

logger = logging.getLogger(__name__)


# Architecture to implementation class mapping
YOLO_ARCHITECTURES = {
    ModelArchitecture.YOLO_V8,
    ModelArchitecture.YOLO_V11,
    ModelArchitecture.YOLO_V12,
    ModelArchitecture.YOLO_POSE,
}


class ModelRegistry:
    """
    Manages model loading and caching.

    Features:
    - Load models on demand
    - Cache loaded models
    - Support A/B testing via version parameter
    - Download models from S3 if not cached locally

    Usage:
        # Initialize with model configs
        registry = ModelRegistry(model_configs)

        # Load a specific model
        model = registry.get_model("yolo_basketball_v1")

        # Load all models for a game
        models = registry.load_models_for_game(["yolo_basketball_v1", "yolo_player_v2"])
    """

    def __init__(
        self,
        model_configs: Optional[List[ModelConfig]] = None,
        cache_dir: str = "./models_cache",
        device: str = "cpu"
    ):
        """
        Initialize the model registry.

        Args:
            model_configs: List of model configurations
            cache_dir: Directory to cache downloaded models
            device: Default device for model loading
        """
        self._configs: Dict[str, ModelConfig] = {}
        self._loaded_models: Dict[str, YOLOModel] = {}
        self.cache_dir = Path(cache_dir)
        self.default_device = device

        # Index configs by model_id
        if model_configs:
            for config in model_configs:
                self._configs[config.model_id] = config

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def register_model(self, config: ModelConfig):
        """
        Register a model configuration.

        Args:
            config: Model configuration to register
        """
        self._configs[config.model_id] = config
        logger.info(f"Registered model: {config.model_id}")

    def register_models(self, configs: List[ModelConfig]):
        """Register multiple model configurations."""
        for config in configs:
            self.register_model(config)

    def get_config(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get model configuration by ID.

        Args:
            model_id: Model identifier

        Returns:
            ModelConfig if found, None otherwise
        """
        return self._configs.get(model_id)

    def get_model(
        self,
        model_id: str,
        device: Optional[str] = None
    ) -> Optional[YOLOModel]:
        """
        Get a loaded model by ID.

        Loads the model if not already loaded.

        Args:
            model_id: Model identifier
            device: Device override (default: use config or registry default)

        Returns:
            Loaded YOLOModel instance, or None if not found
        """
        # Check cache first
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        # Get config
        config = self.get_config(model_id)
        if config is None:
            logger.error(f"Model config not found: {model_id}")
            return None

        # Load model
        model = self._load_model(config, device)
        if model:
            self._loaded_models[model_id] = model

        return model

    def _load_model(
        self,
        config: ModelConfig,
        device: Optional[str] = None
    ) -> Optional[YOLOModel]:
        """
        Load a model from configuration.

        Args:
            config: Model configuration
            device: Device override

        Returns:
            Loaded model or None if failed
        """
        try:
            # Determine device
            model_device = device or self.default_device
            if model_device == "gpu":
                model_device = "cuda:0"

            # Check if architecture is supported
            if config.model_architecture not in YOLO_ARCHITECTURES:
                logger.error(f"Unsupported architecture: {config.model_architecture}")
                return None

            # Ensure model file is available
            model_path = self._ensure_model_available(config)
            if model_path is None:
                return None

            # Create and load model
            half_precision = config.default_params.half_precision if config.default_params else False
            model = YOLOModel(device=model_device, half_precision=half_precision)

            success = model.load(str(model_path))
            if not success:
                logger.error(f"Failed to load model weights: {config.model_id}")
                return None

            # Warmup
            model.warmup()

            logger.info(f"Loaded model: {config.model_id} on {model_device}")
            return model

        except Exception as e:
            logger.error(f"Error loading model {config.model_id}: {e}")
            return None

    def _ensure_model_available(self, config: ModelConfig) -> Optional[Path]:
        """
        Ensure model file is available locally.

        Downloads from URL if not cached.

        Args:
            config: Model configuration

        Returns:
            Path to local model file
        """
        # Determine local path
        url = config.model_url
        filename = url.split("/")[-1]
        local_path = self.cache_dir / filename

        # Check if already cached
        if local_path.exists():
            logger.debug(f"Model cached at: {local_path}")
            return local_path

        # Download if URL
        if url.startswith(("http://", "https://", "s3://")):
            return self._download_model(url, local_path)

        # Check if it's a local path
        if os.path.exists(url):
            return Path(url)

        logger.error(f"Model not found: {url}")
        return None

    def _download_model(self, url: str, local_path: Path) -> Optional[Path]:
        """
        Download model from URL.

        Supports S3 and HTTP(S) URLs.

        Args:
            url: Source URL
            local_path: Local destination path

        Returns:
            Local path if successful
        """
        try:
            if url.startswith("s3://"):
                return self._download_from_s3(url, local_path)
            else:
                return self._download_from_http(url, local_path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None

    def _download_from_s3(self, url: str, local_path: Path) -> Optional[Path]:
        """Download model from S3"""
        try:
            import boto3

            # Parse S3 URL
            parts = url.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            s3 = boto3.client("s3")
            logger.info(f"Downloading model from S3: {url}")
            s3.download_file(bucket, key, str(local_path))

            return local_path

        except ImportError:
            logger.error("boto3 not installed. Cannot download from S3.")
            return None
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return None

    def _download_from_http(self, url: str, local_path: Path) -> Optional[Path]:
        """Download model from HTTP(S)"""
        try:
            import requests

            logger.info(f"Downloading model from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return local_path

        except ImportError:
            logger.error("requests not installed. Cannot download from HTTP.")
            return None
        except Exception as e:
            logger.error(f"HTTP download failed: {e}")
            return None

    def load_models_for_game(
        self,
        model_ids: List[str],
        device: Optional[str] = None
    ) -> Dict[str, YOLOModel]:
        """
        Load all models needed for a game.

        Args:
            model_ids: List of model IDs to load
            device: Device override for all models

        Returns:
            Dict mapping model_id to loaded model
        """
        models = {}

        for model_id in model_ids:
            model = self.get_model(model_id, device=device)
            if model:
                models[model_id] = model
            else:
                logger.warning(f"Failed to load model: {model_id}")

        logger.info(f"Loaded {len(models)}/{len(model_ids)} models for game")
        return models

    def unload_model(self, model_id: str):
        """
        Unload a model from cache.

        Args:
            model_id: Model identifier
        """
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
            logger.info(f"Unloaded model: {model_id}")

    def unload_all(self):
        """Unload all cached models."""
        self._loaded_models.clear()
        logger.info("Unloaded all models")

    def list_registered(self) -> List[str]:
        """List all registered model IDs."""
        return list(self._configs.keys())

    def list_loaded(self) -> List[str]:
        """List all loaded model IDs."""
        return list(self._loaded_models.keys())

    @property
    def registered_count(self) -> int:
        """Number of registered models."""
        return len(self._configs)

    @property
    def loaded_count(self) -> int:
        """Number of loaded models."""
        return len(self._loaded_models)


# =============================================================================
# GLOBAL REGISTRY (optional singleton pattern)
# =============================================================================

_global_registry: Optional[ModelRegistry] = None


def get_global_registry() -> ModelRegistry:
    """
    Get the global model registry instance.

    Creates one if it doesn't exist.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


def init_global_registry(
    model_configs: List[ModelConfig],
    cache_dir: str = "./models_cache",
    device: str = "cpu"
) -> ModelRegistry:
    """
    Initialize the global model registry.

    Args:
        model_configs: Model configurations to register
        cache_dir: Directory for model cache
        device: Default device

    Returns:
        Initialized global registry
    """
    global _global_registry
    _global_registry = ModelRegistry(
        model_configs=model_configs,
        cache_dir=cache_dir,
        device=device
    )
    return _global_registry
