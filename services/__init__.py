"""
Services Module

Contains service implementations for different analysis types:
- Object Detection Service
- Pose Estimation Service

Services self-register with registry on import.
"""

from services.base_service import BaseService, ServiceConfig
from services.registry import register, create, available_types

__all__ = [
    "BaseService",
    "ServiceConfig",
    "register",
    "create",
    "available_types",
]
