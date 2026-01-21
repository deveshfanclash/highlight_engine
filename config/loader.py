"""
Configuration Loader

4-Tier Configuration System:
- Tier 1: ModelRegistry - Pure ML model definitions
- Tier 2: GameTemplate - Sport-specific logic
- Tier 3: DeploymentProfile - Infrastructure configuration
- Tier 4: MatchConfig - Runtime match configuration

Supports loading from:
- YAML files (development/testing)
- MongoDB (production)
- Environment variables (overrides)
"""

import os
import yaml
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from config.schemas.enums import (
    ModelType,
    ModelArchitecture,
    GameCategory,
    MatchStatus,
    InputType,
    OutputFormat,
)
from config.schemas.model import (
    ClassMapping,
    ModelParams,
    ResourceRequirements,
    ModelConfig,
    ModelRegistryConfig,
)
from config.schemas.game import (
    ModelAssignment,
    InferenceSettings,
    GameTemplate,
    ServiceTemplateUnion,
    create_service_template_from_dict,
)
from config.schemas.deployment import (
    DeploymentProfile,
    create_development_profile,
    create_production_profile,
)
from config.schemas.match import (
    MatchConfig,
    MatchOverrides,
    MatchMetadata,
    ResumePosition,
    create_match_config,
)

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Unified configuration loader for the 4-tier system.

    Supports:
    - YAML files for development/testing
    - MongoDB for production
    - Environment variable overrides
    - Caching for performance

    Usage:
        # Development (YAML)
        loader = ConfigLoader()
        game, models = loader.load_from_yaml("config/games/football.yaml")
        deployment = loader.load_deployment_from_yaml("config/deployments/production.yaml")

        # Production (MongoDB)
        loader = ConfigLoader(mongo_uri="mongodb://...")
        game = loader.load_game_template("football")
        deployment = loader.load_deployment_profile("production")
        models = loader.load_models_for_game(game)

        # Create match
        match = loader.create_match(
            match_id="match_123",
            stream_url="https://...",
            game_id="football",
            deployment_profile_id="production",
            overrides={"inference_settings": {"frame_skip": 2}}
        )
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        mongo_db: str = "inference_config",
        cache_enabled: bool = True,
    ):
        """
        Initialize config loader.

        Args:
            mongo_uri: MongoDB connection URI (if None, only YAML loading works)
            mongo_db: MongoDB database name
            cache_enabled: Enable config caching
        """
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.cache_enabled = cache_enabled

        self._mongo_client = None
        self._cache: Dict[str, Any] = {}

    # =========================================================================
    # MONGODB CLIENT
    # =========================================================================

    @property
    def mongo_client(self):
        """Lazy-load MongoDB client"""
        if self._mongo_client is None and self.mongo_uri:
            try:
                from pymongo import MongoClient
                self._mongo_client = MongoClient(self.mongo_uri)
            except ImportError:
                raise ImportError("pymongo is required for MongoDB support")
        return self._mongo_client

    def _get_collection(self, collection_name: str):
        """Get MongoDB collection"""
        if not self.mongo_client:
            raise ValueError("MongoDB URI not configured")
        return self.mongo_client[self.mongo_db][collection_name]

    # =========================================================================
    # TIER 1: MODEL REGISTRY
    # =========================================================================

    @staticmethod
    def _parse_model_config(data: Dict[str, Any]) -> ModelConfig:
        """Parse a single model config from dict"""
        # Parse class mappings
        class_mappings = []
        for cm in data.get("default_class_mapping", []):
            if isinstance(cm, dict):
                class_mappings.append(ClassMapping(**cm))

        # Parse params
        params_data = data.get("default_params", {})
        params = ModelParams(**params_data) if params_data else ModelParams()

        # Parse resources
        resources_data = data.get("resources", {})
        resources = ResourceRequirements(**resources_data) if resources_data else ResourceRequirements()

        return ModelConfig(
            model_id=data["model_id"],
            model_name=data.get("model_name", ""),
            description=data.get("description", ""),
            model_type=ModelType(data["model_type"]),
            model_architecture=ModelArchitecture(
                data.get("model_architecture", "yolov8")
            ),
            model_url=data["model_url"],
            version=data.get("version", "latest"),
            checksum=data.get("checksum"),
            native_classes=data.get("native_classes", []),
            native_class_ids=data.get("native_class_ids", []),
            default_class_mapping=class_mappings,
            default_params=params,
            resources=resources,
            output_format=OutputFormat(data.get("output_format", "bbox")),
            tags=data.get("tags", []),
        )

    @staticmethod
    def _parse_model_registry(models_data: List[Dict[str, Any]]) -> ModelRegistryConfig:
        """Parse model list into ModelRegistryConfig"""
        models = [ConfigLoader._parse_model_config(m) for m in models_data]
        return ModelRegistryConfig(models=models)

    def load_model_registry_from_yaml(self, yaml_path: str) -> ModelRegistryConfig:
        """Load model registry from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        models_data = data.get("models", [data] if "model_id" in data else [])
        return self._parse_model_registry(models_data)

    def load_model_registry(self, game_id: Optional[str] = None) -> ModelRegistryConfig:
        """Load model registry from MongoDB"""
        collection = self._get_collection("models")

        query = {}
        if game_id:
            # Get model IDs from game template first
            game = self.load_game_template(game_id)
            if game:
                model_ids = game.get_model_ids()
                query = {"model_id": {"$in": model_ids}}

        models_data = list(collection.find(query))
        for m in models_data:
            m.pop("_id", None)

        return self._parse_model_registry(models_data)

    # =========================================================================
    # TIER 2: GAME TEMPLATE
    # =========================================================================

    @staticmethod
    def _parse_game_template(data: Dict[str, Any]) -> GameTemplate:
        """Parse game template from dict"""
        # Get game data (might be nested under 'game' key)
        game_data = data.get("game", data)

        # Parse model assignments
        model_assignments = []
        for ma in game_data.get("model_assignments", []):
            model_assignments.append(ModelAssignment(**ma))

        # Parse services
        services = []
        for service_data in game_data.get("services", []):
            service = create_service_template_from_dict(service_data)
            services.append(service)

        # Parse inference settings
        inference_data = game_data.get("inference_settings", {})
        inference_settings = InferenceSettings(**inference_data)

        return GameTemplate(
            game_id=game_data["game_id"],
            game_name=game_data["game_name"],
            game_category=GameCategory(game_data.get("game_category", "ball_sport")),
            description=game_data.get("description", ""),
            model_assignments=model_assignments,
            services=services,
            inference_settings=inference_settings,
            universal_class_definitions=game_data.get("universal_class_definitions", {}),
            config_version=game_data.get("config_version", 1),
            tags=game_data.get("tags", []),
        )

    def load_game_template_from_yaml(self, yaml_path: str) -> GameTemplate:
        """Load game template from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return self._parse_game_template(data)

    def load_game_template(self, game_id: str) -> Optional[GameTemplate]:
        """Load game template from MongoDB"""
        cache_key = f"game:{game_id}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        collection = self._get_collection("game_templates")
        data = collection.find_one({"game_id": game_id})

        if not data:
            return None

        data.pop("_id", None)
        game = self._parse_game_template(data)

        if self.cache_enabled:
            self._cache[cache_key] = game

        return game

    # =========================================================================
    # TIER 3: DEPLOYMENT PROFILE
    # =========================================================================

    @staticmethod
    def _parse_deployment_profile(data: Dict[str, Any]) -> DeploymentProfile:
        """Parse deployment profile from dict"""
        return DeploymentProfile(
            profile_id=data["profile_id"],
            profile_name=data.get("profile_name", ""),
            environment=data.get("environment", "production"),
            local_output_dir=data.get("local_output_dir"),
            aws_region=data.get("aws_region", "us-east-1"),
            db_batch_size=data.get("db_batch_size", 25),
            db_flush_interval_ms=data.get("db_flush_interval_ms", 250),
        )

    def load_deployment_profile_from_yaml(self, yaml_path: str) -> DeploymentProfile:
        """Load deployment profile from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return self._parse_deployment_profile(data)

    def load_deployment_profile(self, profile_id: str) -> Optional[DeploymentProfile]:
        """Load deployment profile from MongoDB or defaults"""
        cache_key = f"deployment:{profile_id}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        # Check for built-in profiles first
        if profile_id == "development":
            return create_development_profile()
        elif profile_id == "production":
            return create_production_profile()

        # Try MongoDB
        if self.mongo_client:
            collection = self._get_collection("deployment_profiles")
            data = collection.find_one({"profile_id": profile_id})

            if data:
                data.pop("_id", None)
                profile = self._parse_deployment_profile(data)

                if self.cache_enabled:
                    self._cache[cache_key] = profile

                return profile

        return None

    # =========================================================================
    # TIER 4: MATCH CONFIG
    # =========================================================================

    def create_match(
        self,
        match_id: str,
        stream_url: str,
        game_id: str,
        deployment_profile_id: str = "production",
        stream_type: str = "hls",
        overrides: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        save_to_db: bool = False,
    ) -> MatchConfig:
        """
        Create a new match configuration.

        Args:
            match_id: Unique match identifier
            stream_url: Stream URL
            game_id: Reference to game template
            deployment_profile_id: Reference to deployment profile
            stream_type: Type of stream (hls, rtsp, mp4)
            overrides: Match-specific overrides
            metadata: Match metadata
            save_to_db: Save to MongoDB if True

        Returns:
            MatchConfig instance
        """
        match = create_match_config(
            match_id=match_id,
            stream_url=stream_url,
            game_id=game_id,
            deployment_profile_id=deployment_profile_id,
            stream_type=InputType(stream_type),
            overrides=overrides,
            metadata=metadata,
        )

        if save_to_db and self.mongo_client:
            collection = self._get_collection("matches")
            collection.insert_one(match.model_dump(mode="json"))

        return match

    def load_match(self, match_id: str) -> Optional[MatchConfig]:
        """Load match config from MongoDB"""
        if not self.mongo_client:
            return None

        collection = self._get_collection("matches")
        data = collection.find_one({"match_id": match_id})

        if not data:
            return None

        data.pop("_id", None)
        return MatchConfig(**data)

    # =========================================================================
    # COMBINED LOADING (Convenience methods)
    # =========================================================================

    @staticmethod
    def load_from_yaml(yaml_path: str) -> Tuple[GameTemplate, ModelRegistryConfig]:
        """
        Load GameTemplate and ModelRegistry from a combined YAML file.

        Expects format:
        ```yaml
        models:
          - model_id: ...
        game_id: ...
        game_name: ...
        services:
          - ...
        ```

        Returns:
            Tuple of (GameTemplate, ModelRegistryConfig)
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse models
        model_registry = ConfigLoader._parse_model_registry(data.get("models", []))

        # Parse game template
        game_template = ConfigLoader._parse_game_template(data)

        return game_template, model_registry

    def load_full_config(
        self,
        game_id: str,
        deployment_profile_id: str = "production",
    ) -> Tuple[GameTemplate, ModelRegistryConfig, DeploymentProfile]:
        """
        Load all three config tiers for a game.

        Returns:
            Tuple of (GameTemplate, ModelRegistryConfig, DeploymentProfile)
        """
        game = self.load_game_template(game_id)
        if not game:
            raise ValueError(f"Game template not found: {game_id}")

        models = self.load_model_registry(game_id)
        deployment = self.load_deployment_profile(deployment_profile_id)
        if not deployment:
            deployment = create_production_profile()

        return game, models, deployment


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_config_from_yaml(yaml_path: str) -> Tuple[GameTemplate, ModelRegistryConfig]:
    """Load game template and models from a combined YAML file"""
    return ConfigLoader.load_from_yaml(yaml_path)


def load_game_from_yaml(yaml_path: str) -> GameTemplate:
    """Load only game template from YAML"""
    game, _ = ConfigLoader.load_from_yaml(yaml_path)
    return game


def load_deployment_from_yaml(yaml_path: str) -> DeploymentProfile:
    """Load deployment profile from YAML"""
    return ConfigLoader().load_deployment_profile_from_yaml(yaml_path)


def load_config_from_env() -> Optional[Tuple[GameTemplate, ModelRegistryConfig]]:
    """
    Load config based on environment variables.

    Expected env vars:
    - CONFIG_SOURCE: 'yaml' or 'mongodb'
    - CONFIG_PATH: Path to YAML file (if yaml)
    - MONGO_URI: MongoDB URI (if mongodb)
    - GAME_ID: Game ID to load
    """
    source = os.getenv("CONFIG_SOURCE", "yaml")

    if source == "yaml":
        path = os.getenv("CONFIG_PATH")
        if not path:
            raise ValueError("CONFIG_PATH env var required for yaml source")
        return ConfigLoader.load_from_yaml(path)

    elif source == "mongodb":
        mongo_uri = os.getenv("MONGO_URI")
        game_id = os.getenv("GAME_ID")
        if not mongo_uri or not game_id:
            raise ValueError("MONGO_URI and GAME_ID env vars required for mongodb source")
        loader = ConfigLoader(mongo_uri=mongo_uri)
        game = loader.load_game_template(game_id)
        models = loader.load_model_registry(game_id)
        return game, models

    else:
        raise ValueError(f"Unknown CONFIG_SOURCE: {source}")


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m config.loader <yaml_path>")
        sys.exit(1)

    yaml_path = sys.argv[1]

    try:
        game_template, model_registry = ConfigLoader.load_from_yaml(yaml_path)

        print("=" * 60)
        print(f"Loaded: {game_template.game_name}")
        print("=" * 60)

        print(f"\nGame ID: {game_template.game_id}")
        print(f"Category: {game_template.game_category}")

        print(f"\nModel Assignments: {len(game_template.model_assignments)}")
        for ma in game_template.model_assignments:
            print(f"  - {ma.model_id} (role: {ma.role})")

        print(f"\nModels in Registry: {len(model_registry.models)}")
        for model in model_registry.models:
            print(f"  - {model.model_id} ({model.model_type})")
            print(f"    URL: {model.model_url}")
            print(f"    Requires GPU: {model.resources.requires_gpu}")

        print(f"\nServices: {len(game_template.services)}")
        for service in game_template.services:
            print(f"  - {service.service_type} (enabled={service.enabled})")

        print("=" * 60)

    except Exception as e:
        print(f"Error loading config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
