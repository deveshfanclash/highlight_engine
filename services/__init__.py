"""
Services Module

Contains service implementations for different analysis types:
- Object Detection Service
- Camera View Service
- HLS Metadata Service
- (Future) Replay Detection Service
- (Future) Transcription Service

Usage:
    from services import create_service_from_config, ServiceRegistry

    # Create service from v2 config
    service = create_service_from_config(game_config, "od_main", match_id, stream_url)

    # Register custom service type
    ServiceRegistry.register("my_service", MyServiceFactory)
"""

from services.base_service import BaseService, ServiceConfig, ServiceRunConfig
from services.registry import (
    ServiceRegistry,
    ServiceFactory,
    create_service,
    create_service_from_config,
    create_all_services,
)

__all__ = [
    # Base
    "BaseService",
    "ServiceConfig",
    "ServiceRunConfig",
    # Registry
    "ServiceRegistry",
    "ServiceFactory",
    "create_service",
    "create_service_from_config",
    "create_all_services",
]
