"""
Pose Estimation Service

Runs pose estimation models on video streams.
Self-registers with ServiceRegistry on import.
"""

from services.pose_service.service import PoseService, PoseServiceConfig, build_pose_config
from config.schemas import ServiceType
from services.registry import register_service

# Self-register with ServiceRegistry
register_service(
    service_type=ServiceType.POSE_ESTIMATION,
    service_class=PoseService,
    config_builder=build_pose_config
)

__all__ = ["PoseService", "PoseServiceConfig", "build_pose_config"]
