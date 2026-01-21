"""
Pose Estimation Service

Runs pose estimation models on video streams.
"""

from services.pose_service.service import PoseService, PoseServiceConfig

__all__ = ["PoseService", "PoseServiceConfig"]
