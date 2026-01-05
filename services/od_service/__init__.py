"""
Object Detection Service

Runs object detection models on video streams.
"""

from services.od_service.service import ODService, ODServiceConfig

__all__ = ["ODService", "ODServiceConfig"]
