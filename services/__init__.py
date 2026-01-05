"""
Services Module

Contains service implementations for different analysis types:
- Object Detection Service
- Camera View Service
- (Future) Replay Detection Service
"""

from services.base_service import BaseService, ServiceConfig

__all__ = [
    "BaseService",
    "ServiceConfig",
]
