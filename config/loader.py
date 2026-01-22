"""
Configuration Loader

Simple loader for game configuration from YAML files.
"""

import yaml
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path

from config.schemas.enums import ModelType, ModelArchitecture, GameCategory
from config.schemas.model import ModelParams, ModelConfig, ModelRegistryConfig
from config.schemas.game import (
    InferenceSettings,
    GameTemplate,
    create_service_template_from_dict,
)

logger = logging.getLogger(__name__)


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

        # Parse inference settings
        inference_data = data.get("inference_settings", {})
        inference_settings = InferenceSettings(**inference_data)

        return GameTemplate(
            game_id=data["game_id"],
            game_name=data["game_name"],
            game_category=GameCategory(data.get("game_category", "ball_sport")),
            services=services,
            inference_settings=inference_settings,
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
            data = yaml.safe_load(f)

        # Parse models
        models = [ConfigLoader._parse_model_config(m) for m in data.get("models", [])]
        model_registry = ModelRegistryConfig(models=models)

        # Parse game
        game_template = ConfigLoader._parse_game_template(data)

        return game_template, model_registry


def load_config(yaml_path: str) -> Tuple[GameTemplate, ModelRegistryConfig]:
    """Load game config from YAML file."""
    return ConfigLoader.load_from_yaml(yaml_path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m config.loader <yaml_path>")
        sys.exit(1)

    game, models = load_config(sys.argv[1])
    print(f"Game: {game.game_name}")
    print(f"Models: {[m.model_id for m in models.models]}")
    print(f"Services: {[s.service_type for s in game.services]}")
