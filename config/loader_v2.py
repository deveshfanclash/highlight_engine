"""
Configuration Loader v2

Loads game configurations from:
- YAML files (development)
- MongoDB (production)

Usage:
    from config.loader_v2 import load_config, load_config_from_mongo

    # Load from YAML
    config = load_config("config/examples/football_v2.yaml")

    # Load from MongoDB
    config = load_config_from_mongo(game_id="football")
"""

import os
import yaml
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from config.schemas_v2 import (
    GameConfig,
    MatchConfig,
    MatchStatus,
    InputConfig,
    ModelConfig,
    ServiceConfig,
    OutputConfig,
    ClassMapping,
)

logger = logging.getLogger(__name__)


# =============================================================================
# YAML LOADING
# =============================================================================

def load_config(path: str) -> GameConfig:
    """
    Load GameConfig from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Validated GameConfig object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If config is invalid
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    return _parse_game_config(data)


def _parse_game_config(data: Dict[str, Any]) -> GameConfig:
    """Parse raw dict into validated GameConfig"""

    # Parse models
    models = []
    for model_data in data.get('models', []):
        # Parse class mappings
        classes = [
            ClassMapping(**c) for c in model_data.get('classes', [])
        ]
        model_data['classes'] = classes
        models.append(ModelConfig(**model_data))

    # Parse services
    services = []
    for service_data in data.get('services', []):
        # Parse input override if present
        if service_data.get('input_override'):
            service_data['input_override'] = InputConfig(**service_data['input_override'])
        services.append(ServiceConfig(**service_data))

    # Parse input config
    input_config = InputConfig(**data.get('input', {}))

    # Parse output config
    output_config = OutputConfig(**data.get('output', {}))

    return GameConfig(
        id=data['id'],
        name=data['name'],
        category=data.get('category', 'ball_sport'),
        input=input_config,
        models=models,
        services=services,
        output=output_config,
        version=data.get('version', 1),
    )


# =============================================================================
# MONGODB LOADING
# =============================================================================

class ConfigStore:
    """
    Configuration store with MongoDB backend.

    Provides:
    - Game config CRUD
    - Match config CRUD
    - Caching (optional)
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        database: str = "inference_db"
    ):
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI")
        self.database = database
        self._client = None

    @property
    def client(self):
        """Lazy-load MongoDB client"""
        if self._client is None:
            if not self.mongo_uri:
                raise ValueError("MongoDB URI not configured")
            try:
                from pymongo import MongoClient
                self._client = MongoClient(self.mongo_uri)
            except ImportError:
                raise ImportError("pymongo required for MongoDB support")
        return self._client

    @property
    def db(self):
        return self.client[self.database]

    # -------------------------------------------------------------------------
    # Game Config
    # -------------------------------------------------------------------------

    def get_game_config(self, game_id: str) -> Optional[GameConfig]:
        """Load game config from MongoDB"""
        collection = self.db['game_configs']
        data = collection.find_one({"id": game_id})

        if not data:
            return None

        data.pop('_id', None)
        return _parse_game_config(data)

    def save_game_config(self, config: GameConfig) -> bool:
        """Save game config to MongoDB"""
        collection = self.db['game_configs']
        data = config.model_dump(mode='json')

        result = collection.update_one(
            {"id": config.id},
            {"$set": data},
            upsert=True
        )
        return result.acknowledged

    def list_game_configs(self) -> list:
        """List all game configs"""
        collection = self.db['game_configs']
        return [
            {"id": doc["id"], "name": doc.get("name")}
            for doc in collection.find({}, {"id": 1, "name": 1})
        ]

    def delete_game_config(self, game_id: str) -> bool:
        """Delete game config"""
        collection = self.db['game_configs']
        result = collection.delete_one({"id": game_id})
        return result.deleted_count > 0

    # -------------------------------------------------------------------------
    # Match Config
    # -------------------------------------------------------------------------

    def create_match(
        self,
        match_id: str,
        game_id: str,
        stream_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MatchConfig:
        """Create new match config"""
        # Verify game exists
        game_config = self.get_game_config(game_id)
        if not game_config:
            raise ValueError(f"Game config not found: {game_id}")

        match = MatchConfig(
            match_id=match_id,
            game_id=game_id,
            stream_url=stream_url,
            status=MatchStatus.PENDING,
            metadata=metadata or {}
        )

        self.save_match(match)
        return match

    def get_match(self, match_id: str) -> Optional[MatchConfig]:
        """Load match config"""
        collection = self.db['matches']
        data = collection.find_one({"match_id": match_id})

        if not data:
            return None

        data.pop('_id', None)
        return MatchConfig(**data)

    def save_match(self, match: MatchConfig) -> bool:
        """Save match config"""
        collection = self.db['matches']
        data = match.model_dump(mode='json')

        result = collection.update_one(
            {"match_id": match.match_id},
            {"$set": data},
            upsert=True
        )
        return result.acknowledged

    def update_match_status(
        self,
        match_id: str,
        status: MatchStatus,
        last_frame: Optional[int] = None
    ) -> bool:
        """Update match status"""
        collection = self.db['matches']

        update = {"status": status.value}
        if last_frame is not None:
            update["last_frame"] = last_frame

        result = collection.update_one(
            {"match_id": match_id},
            {"$set": update}
        )
        return result.modified_count > 0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_config_from_mongo(
    game_id: str,
    mongo_uri: Optional[str] = None
) -> Optional[GameConfig]:
    """
    Load game config from MongoDB.

    Args:
        game_id: Game identifier
        mongo_uri: MongoDB URI (defaults to MONGO_URI env var)

    Returns:
        GameConfig or None if not found
    """
    store = ConfigStore(mongo_uri=mongo_uri)
    return store.get_game_config(game_id)


def load_config_auto(source: str) -> GameConfig:
    """
    Load config from file path or MongoDB game_id.

    Auto-detects:
    - If source ends with .yaml/.yml: Load from file
    - Otherwise: Load from MongoDB by game_id

    Args:
        source: File path or game_id

    Returns:
        GameConfig
    """
    if source.endswith(('.yaml', '.yml')):
        return load_config(source)
    else:
        config = load_config_from_mongo(source)
        if config is None:
            raise ValueError(f"Config not found in MongoDB: {source}")
        return config


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for testing config loading"""
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m config.loader_v2 <config_path>")
        sys.exit(1)

    path = sys.argv[1]

    try:
        config = load_config(path)
        print(f"Loaded config: {config.name}")
        print(f"  ID: {config.id}")
        print(f"  Category: {config.category}")
        print(f"  Models: {len(config.models)}")
        for m in config.models:
            print(f"    - {m.id} ({m.architecture})")
        print(f"  Services: {len(config.services)}")
        for s in config.services:
            print(f"    - {s.id} ({s.type}, device={s.device}, enabled={s.enabled})")
        print("\nValidation: PASSED")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
