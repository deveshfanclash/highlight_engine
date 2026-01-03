"""
Configuration Loader

Loads game configurations from:
- MongoDB (production)
- YAML files (development/testing)
- Environment variables (overrides)

Usage:
    from config.loader import ConfigLoader

    # Load from MongoDB
    loader = ConfigLoader(mongo_uri="mongodb://...")
    config = loader.load_game_config("football")

    # Load from YAML (development)
    config = ConfigLoader.load_from_yaml("config/examples/football_config.yaml")
"""

import os
import yaml
from typing import Optional, Dict, Any
from pathlib import Path

from config.schemas import (
    GameConfig,
    MatchConfig,
    MatchStatus,
    ModelConfig,
    ServiceConfig,
    ClassMapping,
    ModelParams,
    CustomModelConfig,
    ODServiceParams,
    CameraViewServiceParams,
    InferenceSettings,
)


class ConfigLoader:
    """
    Unified configuration loader supporting multiple sources.
    """

    def __init__(self, mongo_uri: Optional[str] = None, mongo_db: str = "inference_db"):
        """
        Initialize config loader.

        Args:
            mongo_uri: MongoDB connection URI (if None, only YAML loading works)
            mongo_db: MongoDB database name
        """
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self._mongo_client = None

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

    # =========================================================================
    # YAML LOADING (Development)
    # =========================================================================

    @staticmethod
    def load_from_yaml(yaml_path: str) -> GameConfig:
        """
        Load GameConfig from a YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            GameConfig object
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return ConfigLoader._parse_game_config(data)

    @staticmethod
    def _parse_game_config(data: Dict[str, Any]) -> GameConfig:
        """Parse raw dict into GameConfig object"""
        # Parse models
        models = []
        for model_data in data.get('models', []):
            # Parse class mapping
            class_mapping = [
                ClassMapping(**cm) for cm in model_data.get('class_mapping', [])
            ]

            # Parse params
            params_data = model_data.get('params', {})
            params = ModelParams(**params_data)

            # Parse custom config if present
            custom_config = None
            if model_data.get('custom_config'):
                custom_config = CustomModelConfig(**model_data['custom_config'])

            models.append(ModelConfig(
                model_id=model_data['model_id'],
                model_type=model_data['model_type'],
                model_architecture=model_data.get('model_architecture', 'yolov8'),
                model_url=model_data['model_url'],
                version=model_data.get('version', 'latest'),
                classes_to_predict=model_data['classes_to_predict'],
                class_mapping=class_mapping,
                params=params,
                output_format=model_data.get('output_format', 'bbox'),
                preferred_device=model_data.get('preferred_device', 'gpu'),
                custom_config=custom_config,
            ))

        # Parse services
        services = []
        for service_data in data.get('services', []):
            service_type = service_data['service_type']
            params_data = service_data.get('params', {})

            # Create appropriate params object based on service type
            if service_type == 'object_detection':
                params = ODServiceParams(**params_data)
            elif service_type == 'camera_view':
                params = CameraViewServiceParams(**params_data)
            else:
                params = params_data  # Generic dict for unknown types

            services.append(ServiceConfig(
                service_type=service_type,
                enabled=service_data.get('enabled', True),
                device=service_data.get('device', 'cpu'),
                params=params,
            ))

        # Parse inference settings
        inference_data = data.get('inference_settings', {})
        inference_settings = InferenceSettings(**inference_data)

        return GameConfig(
            game_id=data['game_id'],
            game_name=data['game_name'],
            game_category=data.get('game_category', 'other'),
            models=models,
            services=services,
            inference_settings=inference_settings,
            config_version=data.get('config_version', 1),
        )

    # =========================================================================
    # MONGODB LOADING (Production)
    # =========================================================================

    def load_game_config(self, game_id: str) -> Optional[GameConfig]:
        """
        Load GameConfig from MongoDB by game_id.

        Args:
            game_id: Game identifier

        Returns:
            GameConfig object or None if not found
        """
        if not self.mongo_client:
            raise ValueError("MongoDB URI not configured")

        db = self.mongo_client[self.mongo_db]
        collection = db['game_configs']

        data = collection.find_one({"game_id": game_id})
        if not data:
            return None

        # Remove MongoDB's _id field
        data.pop('_id', None)

        return self._parse_game_config(data)

    def load_game_config_by_name(self, game_name: str) -> Optional[GameConfig]:
        """Load GameConfig by game name (case-insensitive)"""
        if not self.mongo_client:
            raise ValueError("MongoDB URI not configured")

        db = self.mongo_client[self.mongo_db]
        collection = db['game_configs']

        data = collection.find_one({
            "game_name": {"$regex": f"^{game_name}$", "$options": "i"}
        })
        if not data:
            return None

        data.pop('_id', None)
        return self._parse_game_config(data)

    def save_game_config(self, config: GameConfig) -> bool:
        """
        Save GameConfig to MongoDB.

        Args:
            config: GameConfig object to save

        Returns:
            True if successful
        """
        if not self.mongo_client:
            raise ValueError("MongoDB URI not configured")

        db = self.mongo_client[self.mongo_db]
        collection = db['game_configs']

        # Convert to dict
        data = config.model_dump()

        # Upsert by game_id
        result = collection.update_one(
            {"game_id": config.game_id},
            {"$set": data},
            upsert=True
        )

        return result.acknowledged

    # =========================================================================
    # MATCH CONFIG OPERATIONS
    # =========================================================================

    def create_match_config(
        self,
        match_id: str,
        game_id: str,
        stream_url: str,
        stream_type: str = "hls",
        **kwargs
    ) -> MatchConfig:
        """
        Create a new match config with frozen game config snapshot.

        Args:
            match_id: Unique match identifier
            game_id: Game ID to load config from
            stream_url: Stream URL for this match
            stream_type: Type of stream (hls, rtsp, mp4)
            **kwargs: Additional match metadata (league, tournament, etc.)

        Returns:
            MatchConfig with frozen config snapshot
        """
        # Load game config
        game_config = self.load_game_config(game_id)
        if not game_config:
            raise ValueError(f"Game config not found for game_id: {game_id}")

        # Create match config with frozen snapshot
        match_config = MatchConfig(
            match_id=match_id,
            game_id=game_id,
            stream_url=stream_url,
            stream_type=stream_type,
            status=MatchStatus.PENDING,
            config_snapshot=game_config,
            **kwargs
        )

        return match_config

    def save_match_config(self, config: MatchConfig) -> bool:
        """Save MatchConfig to MongoDB"""
        if not self.mongo_client:
            raise ValueError("MongoDB URI not configured")

        db = self.mongo_client[self.mongo_db]
        collection = db['matches']

        data = config.model_dump()

        result = collection.update_one(
            {"match_id": config.match_id},
            {"$set": data},
            upsert=True
        )

        return result.acknowledged

    def load_match_config(self, match_id: str) -> Optional[MatchConfig]:
        """Load MatchConfig from MongoDB"""
        if not self.mongo_client:
            raise ValueError("MongoDB URI not configured")

        db = self.mongo_client[self.mongo_db]
        collection = db['matches']

        data = collection.find_one({"match_id": match_id})
        if not data:
            return None

        data.pop('_id', None)

        # Parse the nested config_snapshot
        if data.get('config_snapshot'):
            data['config_snapshot'] = self._parse_game_config(data['config_snapshot'])

        return MatchConfig(**data)

    def update_match_status(
        self,
        match_id: str,
        status: MatchStatus,
        last_frame: Optional[int] = None
    ) -> bool:
        """Update match status"""
        if not self.mongo_client:
            raise ValueError("MongoDB URI not configured")

        db = self.mongo_client[self.mongo_db]
        collection = db['matches']

        update_data = {"status": status.value}
        if last_frame is not None:
            update_data["last_processed_frame"] = last_frame

        result = collection.update_one(
            {"match_id": match_id},
            {"$set": update_data}
        )

        return result.modified_count > 0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_config_from_yaml(yaml_path: str) -> GameConfig:
    """Convenience function to load config from YAML"""
    return ConfigLoader.load_from_yaml(yaml_path)


def load_config_from_env() -> Optional[GameConfig]:
    """
    Load config based on environment variables.

    Expected env vars:
    - CONFIG_SOURCE: 'yaml' or 'mongodb'
    - CONFIG_PATH: Path to YAML file (if yaml)
    - MONGO_URI: MongoDB URI (if mongodb)
    - GAME_ID: Game ID to load
    """
    source = os.getenv('CONFIG_SOURCE', 'yaml')

    if source == 'yaml':
        path = os.getenv('CONFIG_PATH')
        if not path:
            raise ValueError("CONFIG_PATH env var required for yaml source")
        return ConfigLoader.load_from_yaml(path)

    elif source == 'mongodb':
        mongo_uri = os.getenv('MONGO_URI')
        game_id = os.getenv('GAME_ID')
        if not mongo_uri or not game_id:
            raise ValueError("MONGO_URI and GAME_ID env vars required for mongodb source")
        loader = ConfigLoader(mongo_uri=mongo_uri)
        return loader.load_game_config(game_id)

    else:
        raise ValueError(f"Unknown CONFIG_SOURCE: {source}")


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m config.loader <yaml_path>")
        sys.exit(1)

    yaml_path = sys.argv[1]

    try:
        config = ConfigLoader.load_from_yaml(yaml_path)
        print("=" * 60)
        print(f"Loaded config for: {config.game_name}")
        print("=" * 60)
        print(f"Game ID: {config.game_id}")
        print(f"Category: {config.game_category}")
        print(f"Models: {len(config.models)}")
        for model in config.models:
            print(f"  - {model.model_id} ({model.model_type})")
            print(f"    Classes: {[cm.universal_class_name for cm in model.class_mapping]}")
        print(f"Services: {len(config.services)}")
        for service in config.services:
            print(f"  - {service.service_type} (enabled={service.enabled}, device={service.device})")
        print("=" * 60)
        print("\nJSON Export:")
        print(json.dumps(config.model_dump(), indent=2, default=str))
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
