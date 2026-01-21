"""
Services Module

Contains service implementations for different analysis types:
- Object Detection Service
- Pose Estimation Service
- (Future) Replay Detection Service

Services self-register with ServiceRegistry on import.
"""

from services.base_service import BaseService, ServiceConfig
from services.registry import ServiceRegistry, register_service

__all__ = [
    "BaseService",
    "ServiceConfig",
    "ServiceRegistry",
    "register_service",
]
