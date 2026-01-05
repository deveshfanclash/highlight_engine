"""
Camera View Service

Detects camera cuts/view changes in video streams using perceptual hashing.
"""

from services.camera_view_service.service import CameraViewService, CameraViewServiceConfig

__all__ = ["CameraViewService", "CameraViewServiceConfig"]
