"""
Configuration Loader

Loads game and model configurations from:
- MongoDB (production)
- YAML files (development/testing)
- Environment variables (overrides)

New Structure:
- GameConfig: References model_ids (not embedding models)
- ModelRegistryConfig: Independent model definitions
- Typed service configs per service type
"""

import os
import yaml
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from config.schemas import (
    # Enums
    ModelType,
    ModelArchitecture,
    ServiceType,
    GameCategory,
    MatchStatus,
    InputType,
    OutputFormat,
    # Model
    ClassMapping,
    ModelParams,
    ModelConfig,
    ModelRegistryConfig,
    # Service
    ODServiceConfig,
    CameraViewServiceConfig,
    SegmentationServiceConfig,
    ReplayDetectionServiceConfig,
    EventDetectionServiceConfig,
    AudioAnalysisServiceConfig,
    HLSMetadataServiceConfig,
    create_service_from_dict,
    # Game
    InferenceSettings,
    GameConfig,
    MatchConfig,
)


class ConfigLoader:
    """
    Unified configuration loader supporting multiple sources.

    New structure separates:
    - GameConfig: What services to run, references model_ids
    - ModelRegistryConfig: Independent model definitions
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
    def load_from_yaml(yaml_path: str) -> Tuple[GameConfig, ModelRegistryConfig]:
        """
        Load GameConfig and ModelRegistryConfig from a YAML file.

        The YAML file should have two top-level keys:
        - game: GameConfig data
        - models: List of ModelConfig data

        Args:
            yaml_path: Path to YAML config file

        Returns:
            Tuple of (GameConfig, ModelRegistryConfig)
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse models first (they're independent)
        model_registry = ConfigLoader._parse_model_registry(data.get('models', []))

        # Parse game config (references model_ids)
        game_config = ConfigLoader._parse_game_config(data)

        return game_config, model_registry

    @staticmethod
    def load_game_from_yaml(yaml_path: str) -> GameConfig:
        """Load only GameConfig from YAML (backward compatibility)"""
        game_config, _ = ConfigLoader.load_from_yaml(yaml_path)
        return game_config

    @staticmethod
    def _parse_model_registry(models_data: List[Dict[str, Any]]) -> ModelRegistryConfig:
        """Parse model list into ModelRegistryConfig"""
        models = []

        for model_data in models_data:
            # Parse class mapping
            class_mapping = []
            for cm in model_data.get('class_mapping', []):
                if isinstance(cm, dict):
                    class_mapping.append(ClassMapping(
                        model_class_id=cm['model_class_id'],
                        universal_class_name=cm['universal_class_name'],
                        confidence_threshold=cm.get('confidence_threshold'),
                    ))

            # Parse params
            params_data = model_data.get('params', {})
            params = ModelParams(
                confidence_threshold=params_data.get('confidence_threshold', 0.5),
                iou_threshold=params_data.get('iou_threshold', 0.45),
                max_detections=params_data.get('max_detections', 100),
                batch_size=params_data.get('batch_size', 1),
                input_size=params_data.get('input_size'),
                half_precision=params_data.get('half_precision', False),
                extra=params_data.get('extra', {}),
            )

            models.append(ModelConfig(
                model_id=model_data['model_id'],
                model_name=model_data.get('model_name', ''),
                model_type=ModelType(model_data['model_type']),
                model_architecture=ModelArchitecture(
                    model_data.get('model_architecture', 'yolov8')
                ),
                model_url=model_data['model_url'],
                version=model_data.get('version', 'latest'),
                classes_to_predict=model_data.get('classes_to_predict', []),
                class_mapping=class_mapping,
                params=params,
                output_format=OutputFormat(model_data.get('output_format', 'bbox')),
                preferred_device=model_data.get('preferred_device', 'gpu'),
            ))

        return ModelRegistryConfig(models=models)

    @staticmethod
    def _parse_game_config(data: Dict[str, Any]) -> GameConfig:
        """Parse raw dict into GameConfig object"""
        # Get game data (might be nested under 'game' key or at root)
        game_data = data.get('game', data)

        # Extract model_ids from services or models list
        model_ids = game_data.get('model_ids', [])

        # If model_ids not provided, extract from models list (legacy support)
        if not model_ids and 'models' in data:
            model_ids = [m.get('model_id') for m in data['models'] if m.get('model_id')]

        # Parse services
        services = []
        for service_data in game_data.get('services', []):
            service = create_service_from_dict(service_data)
            services.append(service)

        # Parse inference settings
        inference_data = game_data.get('inference_settings', {})
        inference_settings = InferenceSettings(
            target_fps=inference_data.get('target_fps', 25),
            frame_skip=inference_data.get('frame_skip', 1),
            processing_resolution=inference_data.get('processing_resolution', [1280, 720]),
            stream_buffer_size=inference_data.get('stream_buffer_size', 30),
        )

        return GameConfig(
            game_id=game_data['game_id'],
            game_name=game_data['game_name'],
            game_category=GameCategory(game_data.get('game_category', 'ball_sport')),
            model_ids=model_ids,
            services=services,
            inference_settings=inference_settings,
            config_version=game_data.get('config_version', 1),
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

    def load_model_registry(self, game_id: Optional[str] = None) -> ModelRegistryConfig:
        """
        Load ModelRegistryConfig from MongoDB.

        Args:
            game_id: If provided, only load models for this game

        Returns:
            ModelRegistryConfig with all relevant models
        """
        if not self.mongo_client:
            raise ValueError("MongoDB URI not configured")

        db = self.mongo_client[self.mongo_db]
        collection = db['models']

        # Query based on game_id if provided
        query = {}
        if game_id:
            # First get game config to find model_ids
            game_config = self.load_game_config(game_id)
            if game_config and game_config.model_ids:
                query = {"model_id": {"$in": game_config.model_ids}}

        models_data = list(collection.find(query))

        # Remove MongoDB _id fields
        for model in models_data:
            model.pop('_id', None)

        return self._parse_model_registry(models_data)

    def load_game_with_models(
        self,
        game_id: str
    ) -> Tuple[Optional[GameConfig], ModelRegistryConfig]:
        """
        Load both GameConfig and its associated models.

        Args:
            game_id: Game identifier

        Returns:
            Tuple of (GameConfig, ModelRegistryConfig)
        """
        game_config = self.load_game_config(game_id)
        model_registry = self.load_model_registry(game_id)
        return game_config, model_registry

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
        Create a new match config.

        Args:
            match_id: Unique match identifier
            game_id: Game ID to reference
            stream_url: Stream URL for this match
            stream_type: Type of stream (hls, rtsp, mp4)
            **kwargs: Additional match metadata (league, tournament, etc.)

        Returns:
            MatchConfig instance
        """
        match_config = MatchConfig(
            match_id=match_id,
            game_id=game_id,
            stream_url=stream_url,
            stream_type=InputType(stream_type),
            status=MatchStatus.PENDING,
            **kwargs
        )

        return match_config

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

        return MatchConfig(**data)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_config_from_yaml(yaml_path: str) -> Tuple[GameConfig, ModelRegistryConfig]:
    """Convenience function to load config from YAML"""
    return ConfigLoader.load_from_yaml(yaml_path)


def load_game_from_yaml(yaml_path: str) -> GameConfig:
    """Convenience function to load only game config from YAML"""
    return ConfigLoader.load_game_from_yaml(yaml_path)


def load_config_from_env() -> Optional[Tuple[GameConfig, ModelRegistryConfig]]:
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
        return loader.load_game_with_models(game_id)

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
        game_config, model_registry = ConfigLoader.load_from_yaml(yaml_path)
        print("=" * 60)
        print(f"Loaded config for: {game_config.game_name}")
        print("=" * 60)
        print(f"Game ID: {game_config.game_id}")
        print(f"Category: {game_config.game_category}")
        print(f"Model IDs: {game_config.model_ids}")
        print(f"\nModels in Registry: {len(model_registry.models)}")
        for model in model_registry.models:
            print(f"  - {model.model_id} ({model.model_type})")
            classes = [cm.universal_class_name for cm in model.class_mapping]
            print(f"    Classes: {classes}")
        print(f"\nServices: {len(game_config.services)}")
        for service in game_config.services:
            print(f"  - {service.service_type} (enabled={service.enabled}, device={service.device})")
        print("=" * 60)
        print("\nGame Config JSON:")
        print(json.dumps(game_config.model_dump(), indent=2, default=str))
    except Exception as e:
        print(f"Error loading config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
