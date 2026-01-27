"""
Infrastructure Configuration

Centralized environment-based configuration for infrastructure services.
All infrastructure params (AWS, Kafka, Slack, logging, paths) are loaded from
environment variables with sensible defaults.

Usage:
    from config.environment import get_infra_config

    # Normal mode (reads from os.getenv)
    config = get_infra_config()

    # Local mode (no infrastructure dependencies)
    config = get_infra_config(local_mode=True)
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Project root (where this file's parent directory is)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class InfraConfig:
    """
    Infrastructure configuration loaded from environment variables.

    In local_mode, all infrastructure features are disabled and
    no environment variables are required.
    """

    # Mode
    local_mode: bool = False

    # AWS Configuration
    aws_region: str = "us-east-1"
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None

    # Kafka Configuration
    kafka_broker_url: Optional[str] = None
    kafka_topic: Optional[str] = None

    # Slack Configuration
    slack_webhook_url: Optional[str] = None

    # Logging Configuration
    log_file_name: str = "inference"
    log_group_name: Optional[str] = None  # CloudWatch log group
    log_level: str = "INFO"

    # Paths (dynamic based on project root)
    base_path: Path = field(default_factory=lambda: PROJECT_ROOT/"runs")
    models_cache_path: Path = field(default_factory=lambda: PROJECT_ROOT / "models_cache")
    logs_path: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    output_path: Path = field(default_factory=lambda: PROJECT_ROOT / "output")

    def __post_init__(self):
        """Ensure paths are Path objects and create directories if needed."""
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)
        if isinstance(self.models_cache_path, str):
            self.models_cache_path = Path(self.models_cache_path)
        if isinstance(self.logs_path, str):
            self.logs_path = Path(self.logs_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

    @property
    def has_aws_credentials(self) -> bool:
        """Check if AWS credentials are configured."""
        return bool(self.aws_access_key and self.aws_secret_key)

    @property
    def has_kafka(self) -> bool:
        """Check if Kafka is configured."""
        return bool(self.kafka_broker_url and self.kafka_topic)

    @property
    def has_slack(self) -> bool:
        """Check if Slack is configured."""
        return bool(self.slack_webhook_url)

    @property
    def can_use_dynamodb(self) -> bool:
        """Check if DynamoDB can be used (not local mode and has credentials or IAM role)."""
        return not self.local_mode

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.models_cache_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)


def _load_from_env() -> InfraConfig:
    #TODO: load from dotenv if env is not registered througha container or shell set with env.
    """Load infrastructure config from environment variables."""
    return InfraConfig(
        local_mode=False,

        # AWS
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key=os.getenv("AWS_ACCESS_KEY_DB"),
        aws_secret_key=os.getenv("AWS_SECRET_KEY_DB"),

        # Slack
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),

        # Logging
        log_file_name=os.getenv("LOG_FILE_NAME", "inference"),
        log_group_name=os.getenv("LOG_GROUP_NAME"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),

        # Paths
        base_path=Path(os.getenv("BASE_PATH", str(PROJECT_ROOT))),
        models_cache_path=Path(os.getenv("MODELS_CACHE_PATH", str(PROJECT_ROOT / "models_cache"))),
        logs_path=Path(os.getenv("LOGS_PATH", str(PROJECT_ROOT / "logs"))),
        output_path=Path(os.getenv("OUTPUT_PATH", str(PROJECT_ROOT / "output"))),
    )


def _create_local_config(base_path: Optional[str] = None) -> InfraConfig:
    """Create a local-mode config that works without any environment variables."""
    base_path_resolved = Path(base_path) if base_path else PROJECT_ROOT

    return InfraConfig(
        local_mode=True,

        # AWS - disabled
        aws_region="us-east-1",
        aws_access_key=None,
        aws_secret_key=None,

        # Slack - disabled
        slack_webhook_url=None,

        # Logging - local only
        log_file_name="inference",
        log_group_name=None,
        log_level="INFO",

        # Paths
        base_path=base_path_resolved,
        models_cache_path=base_path_resolved / "models_cache",
        logs_path=base_path_resolved / "logs",
        output_path=base_path_resolved / "output",
    )


# Global config instance (singleton pattern)
_config: Optional[InfraConfig] = None


def get_infra_config(
    local_mode: bool = False,
    base_path: Optional[str] = None,
    force_reload: bool = False
) -> InfraConfig:
    """
    Get the infrastructure configuration.

    Args:
        local_mode: If True, return config for local mode (no infrastructure)
        base_path: Custom base path directory (only used in local mode)
        force_reload: If True, reload config even if already loaded

    Returns:
        InfraConfig instance
    """
    global _config

    if _config is not None and not force_reload and not local_mode:
        return _config

    if local_mode:
        config = _create_local_config(base_path)
    else:
        config = _load_from_env()

    # Ensure directories exist
    config.ensure_directories()

    # Cache if not local mode
    if not local_mode:
        _config = config

    return config


def reset_config():
    """Reset the global config (useful for testing)."""
    global _config
    _config = None
