"""
Object Detection Service

Runs object detection models on video streams.
Self-registers with ServiceRegistry on import.
"""

from services.od_service.service import ODService, ODServiceConfig, build_od_config
from config.schemas import ServiceType
from services.registry import register_service

# Self-register with ServiceRegistry
register_service(
    service_type=ServiceType.OBJECT_DETECTION,
    service_class=ODService,
    config_builder=build_od_config
)

__all__ = ["ODService", "ODServiceConfig", "build_od_config"]
