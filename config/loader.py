"""
Configuration Loader

Simple loader for game configuration from YAML files.
Provides both low-level loading and high-level resolved service contexts.
"""

import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from config.schemas.enums import ModelType, ModelArchitecture, GameCategory, ServiceType
from config.schemas.model import ModelParams, ModelConfig, ModelRegistryConfig
from config.schemas.game import (
    Defaults,
    InferenceDefaults,
    OutputDefaults,
    GameTemplate,
    create_service_template_from_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class ResolvedServiceConfig:
    """
    Fully resolved service configuration.

    Contains all settings needed to create and run a service,
    with defaults and overrides already merged.
    """
    # Service identity
    service_type: ServiceType
    service_id: str

    # Model configuration (flat dict)
    model_config: Dict[str, Any]

    # Resolved settings (defaults + overrides merged)
    inference_settings: Dict[str, Any]
    output_settings: Dict[str, Any]

    # Device
    device: str = "cuda:0"

    # Additional service-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    """Load game configuration from YAML files."""

    @staticmethod
    def _parse_model_config(data: Dict[str, Any]) -> ModelConfig:
        """Parse a single model config."""
        params_data = data.get("default_params", {})
        params = ModelParams(**params_data) if params_data else ModelParams()

        return ModelConfig(
            model_id=data["model_id"],
            model_name=data.get("model_name", ""),
            model_type=ModelType(data.get("model_type", "object_detection")),
            model_architecture=ModelArchitecture(data.get("model_architecture", "yolov8")),
            model_url=data.get("model_url", ""),
            model_path=data.get("model_path", ""),
            classes_to_detect=data.get("classes_to_detect", []),
            default_params=params,
        )

    @staticmethod
    def _parse_game_template(data: Dict[str, Any]) -> GameTemplate:
        """Parse game template."""
        # Parse services (each service now has model_id directly)
        services = []
        for svc in data.get("services", []):
            services.append(create_service_template_from_dict(svc))

        # Parse defaults (new hybrid structure)
        defaults_data = data.get("defaults", {})
        defaults = Defaults(
            inference=InferenceDefaults(**defaults_data.get("inference", {})),
            output=OutputDefaults(**defaults_data.get("output", {})),
        )

        return GameTemplate(
            game_id=data["game_id"],
            game_name=data["game_name"],
            game_category=GameCategory(data.get("game_category", "ball_sport")),
            services=services,
            defaults=defaults,
        )

    @staticmethod
    def load_from_yaml(yaml_path: str) -> Tuple[GameTemplate, ModelRegistryConfig]:
        """
        Load game config and models from YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            Tuple of (GameTemplate, ModelRegistryConfig)
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {yaml_path}")

        with open(path, 'r') as f:
            data: dict[list[dict[str, Any]]] = yaml.safe_load(f)

        # Parse models
        models = [ConfigLoader._parse_model_config(m) for m in data.get("models", [])]
        model_registry = ModelRegistryConfig(models=models)

        # Parse game
        game_template = ConfigLoader._parse_game_template(data)

        return game_template, model_registry


def load_config(yaml_path: str) -> Tuple[GameTemplate, ModelRegistryConfig]:
    """Load game config from YAML file."""
    return ConfigLoader.load_from_yaml(yaml_path)


def load_service_configs(yaml_path: str) -> List[ResolvedServiceConfig]:
    """
    Load and resolve all service configurations from YAML.

    This is the recommended high-level API. Returns fully resolved
    configs with defaults and overrides already merged.

    Args:
        yaml_path: Path to game config YAML file

    Returns:
        List of ResolvedServiceConfig, one per enabled service
    """
    game_template, model_registry = ConfigLoader.load_from_yaml(yaml_path)

    resolved_configs = []
    for service in game_template.get_enabled_services():
        # Get model for this service
        model_id = getattr(service, 'model_id', None)
        if not model_id:
            logger.warning(f"Service {service.service_type.value} has no model_id, skipping")
            continue

        model = model_registry.get_model(model_id)
        if not model:
            logger.warning(f"Model '{model_id}' not found, skipping service")
            continue

        # Resolve settings (merge defaults + overrides)
        inference_settings = game_template.resolve_inference_settings(service)
        output_settings = game_template.resolve_output_settings(service)

        # Build service ID
        service_id = f"{service.service_type.value}_{model_id}"

        # Create resolved config
        resolved = ResolvedServiceConfig(
            service_type=service.service_type,
            service_id=service_id,
            model_config=model.model_dump(),
            inference_settings=inference_settings,
            output_settings=output_settings,
            device=service.device,
            extra={
                "target_width": getattr(service, 'target_width', None),
                "target_height": getattr(service, 'target_height', None),
            }
        )
        resolved_configs.append(resolved)

    return resolved_configs


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m config.loader <yaml_path>")
        sys.exit(1)

    # Test both APIs
    game, models = load_config(sys.argv[1])
    print(f"Game: {game.game_name}")
    print(f"Models: {[m.model_id for m in models.models]}")
    print(f"Services: {[s.service_type for s in game.services]}")

    print("\n--- Resolved Configs ---")
    configs = load_service_configs(sys.argv[1])
    for cfg in configs:
        print(f"  {cfg.service_id}: {cfg.service_type.value} on {cfg.device}")
