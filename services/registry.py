"""
Service Registry

Registry pattern for creating services by type string.
Allows adding new service types without code changes.

Usage:
    from services.registry import ServiceRegistry, create_service_from_config

    # Register a custom service
    ServiceRegistry.register("my_service", MyService, MyServiceConfig)

    # Create service from v2 config
    service = create_service_from_config(game_config, service_id, match_config)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Type, Optional, Any, List, Tuple

from services.base_service import BaseService, ServiceRunConfig

logger = logging.getLogger(__name__)


# =============================================================================
# SERVICE FACTORY INTERFACE
# =============================================================================

class ServiceFactory(ABC):
    """
    Abstract factory for creating services.

    Implement this to define how a service is created from v2 config.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        match_id: str,
        stream_url: str,
        service_config: "ServiceConfig",
        game_config: "GameConfig",
        **kwargs
    ) -> BaseService:
        """
        Create a service instance.

        Args:
            match_id: Match identifier
            stream_url: Stream URL
            service_config: ServiceConfig from schemas_v2
            game_config: Full GameConfig (for model lookup etc.)
            **kwargs: Additional parameters (start_frame, start_segment, etc.)

        Returns:
            Configured service instance (not yet running)
        """
        pass


# =============================================================================
# BUILT-IN FACTORIES
# =============================================================================

class ObjectDetectionFactory(ServiceFactory):
    """Factory for Object Detection services"""

    @classmethod
    def create(
        cls,
        match_id: str,
        stream_url: str,
        service_config: "ServiceConfig",
        game_config: "GameConfig",
        **kwargs
    ) -> BaseService:
        from services.od_service.service import ODService, ODServiceConfig

        # Get model config for this service
        model_configs = []
        for model_id in service_config.models:
            model_config = game_config.get_model(model_id)
            if model_config:
                model_configs.append(model_config)

        if not model_configs:
            raise ValueError(f"No models found for service {service_config.id}")

        # Use first model (OD service processes one model at a time)
        model_config = model_configs[0]

        # Build class mapping
        class_mapping = {}
        for cls_map in model_config.classes:
            class_mapping[cls_map.model_class] = cls_map.name

        # Get input config (service override or global)
        input_config = service_config.input_override or game_config.input

        # Get resolution
        resolution = input_config.resolution or [1280, 720]

        # Create OD service config
        od_config = ODServiceConfig(
            match_id=match_id,
            service_id=service_config.id,
            stream_url=stream_url,
            stream_type=input_config.source_type,
            target_width=resolution[0],
            target_height=resolution[1],
            frame_skip=input_config.frame_skip,
            start_frame=kwargs.get("start_frame", input_config.start_frame),
            start_segment=kwargs.get("start_segment", input_config.start_segment),
            device=service_config.device,
            db_table_name=game_config.output.primary.get("table", "inference_results"),
            db_batch_size=game_config.output.batch_size,
            db_flush_interval_ms=game_config.output.flush_interval_ms,
            local_output_dir=kwargs.get("local_output_dir"),
            model_id=model_config.id,
            model_url=model_config.weights,
            model_architecture=model_config.architecture,
            confidence_threshold=model_config.params.get("confidence", 0.5),
            iou_threshold=model_config.params.get("iou", 0.45),
            max_detections=model_config.params.get("max_detections", 100),
            batch_size=model_config.params.get("batch_size", 1),
            half_precision=model_config.params.get("half_precision", False),
            class_mapping=class_mapping,
        )

        return ODService(od_config)


class CameraViewFactory(ServiceFactory):
    """Factory for Camera View services"""

    @classmethod
    def create(
        cls,
        match_id: str,
        stream_url: str,
        service_config: "ServiceConfig",
        game_config: "GameConfig",
        **kwargs
    ) -> BaseService:
        from services.camera_view_service.service import CameraViewService, CameraViewServiceConfig

        # Get input config (service override or global)
        input_config = service_config.input_override or game_config.input

        # Camera view often uses lower resolution
        resolution = input_config.resolution or [640, 480]

        # Get service params
        params = service_config.params or {}

        config = CameraViewServiceConfig(
            match_id=match_id,
            service_id=service_config.id,
            stream_url=stream_url,
            stream_type=input_config.source_type,
            target_width=resolution[0],
            target_height=resolution[1],
            frame_skip=input_config.frame_skip,
            start_frame=kwargs.get("start_frame", input_config.start_frame),
            start_segment=kwargs.get("start_segment", input_config.start_segment),
            device=service_config.device,
            db_table_name=game_config.output.primary.get("table", "inference_results"),
            db_batch_size=game_config.output.batch_size,
            db_flush_interval_ms=game_config.output.flush_interval_ms,
            local_output_dir=kwargs.get("local_output_dir"),
            phash_threshold=params.get("phash_threshold", 20),
            histogram_threshold=params.get("histogram_threshold", 0.90),
            min_frame_gap=params.get("min_frame_gap", 25),
        )

        return CameraViewService(config)


class HLSMetadataFactory(ServiceFactory):
    """Factory for HLS Metadata services"""

    @classmethod
    def create(
        cls,
        match_id: str,
        stream_url: str,
        service_config: "ServiceConfig",
        game_config: "GameConfig",
        **kwargs
    ) -> BaseService:
        from services.hls_metadata_service.service import HLSMetadataService, HLSMetadataServiceConfig

        params = service_config.params or {}

        config = HLSMetadataServiceConfig(
            match_id=match_id,
            stream_url=stream_url,
            db_table_name=game_config.output.primary.get("table", "video_frames_metadata"),
            local_output_dir=kwargs.get("local_output_dir"),
            poll_interval=params.get("poll_interval", 5),
            timeout_no_segment=params.get("timeout_no_segment", 60),
        )

        return HLSMetadataService(config)


class ReplayDetectionFactory(ServiceFactory):
    """Factory for Replay Detection services (placeholder)"""

    @classmethod
    def create(
        cls,
        match_id: str,
        stream_url: str,
        service_config: "ServiceConfig",
        game_config: "GameConfig",
        **kwargs
    ) -> BaseService:
        # Replay detection not yet implemented
        raise NotImplementedError(
            "Replay detection service not yet implemented. "
            "Register a custom service with ServiceRegistry.register()"
        )


class TranscriptionFactory(ServiceFactory):
    """Factory for Audio Transcription services (placeholder)"""

    @classmethod
    def create(
        cls,
        match_id: str,
        stream_url: str,
        service_config: "ServiceConfig",
        game_config: "GameConfig",
        **kwargs
    ) -> BaseService:
        # Audio transcription not yet implemented
        raise NotImplementedError(
            "Transcription service not yet implemented. "
            "Register a custom service with ServiceRegistry.register()"
        )


class CustomFactory(ServiceFactory):
    """
    Factory for custom services specified via module/class path.

    The service_config.params must include:
    - module: Python module path (e.g., "my_services.custom")
    - class_name: Service class name (e.g., "MyCustomService")
    - config_class: Optional config class name (defaults to ServiceRunConfig)
    """

    @classmethod
    def create(
        cls,
        match_id: str,
        stream_url: str,
        service_config: "ServiceConfig",
        game_config: "GameConfig",
        **kwargs
    ) -> BaseService:
        import importlib

        params = service_config.params
        if not params:
            raise ValueError("Custom service requires 'module' and 'class_name' in params")

        module_path = params.get("module")
        class_name = params.get("class_name")

        if not module_path or not class_name:
            raise ValueError("Custom service params must include 'module' and 'class_name'")

        # Import the module and class
        try:
            module = importlib.import_module(module_path)
            service_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {class_name} from {module_path}: {e}")

        # Get config class if specified
        config_class_name = params.get("config_class")
        if config_class_name:
            config_class = getattr(module, config_class_name)
        else:
            config_class = ServiceRunConfig

        # Get input config
        input_config = service_config.input_override or game_config.input
        resolution = input_config.resolution or [1280, 720]

        # Create config
        config = config_class(
            match_id=match_id,
            service_id=service_config.id,
            stream_url=stream_url,
            stream_type=input_config.source_type,
            target_width=resolution[0],
            target_height=resolution[1],
            frame_skip=input_config.frame_skip,
            device=service_config.device,
            start_frame=kwargs.get("start_frame", input_config.start_frame),
            start_segment=kwargs.get("start_segment", input_config.start_segment),
            db_table_name=game_config.output.primary.get("table", "inference_results"),
            local_output_dir=kwargs.get("local_output_dir"),
            params=params,  # Pass all params for custom processing
        )

        return service_class(config)


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

class ServiceRegistry:
    """
    Registry for service types.

    Maps service type strings to factory classes.
    Add new service types by calling register().
    """

    _factories: Dict[str, Type[ServiceFactory]] = {}

    @classmethod
    def register(
        cls,
        service_type: str,
        factory: Type[ServiceFactory]
    ) -> None:
        """
        Register a service factory for a service type.

        Args:
            service_type: Service type name (e.g., "object_detection")
            factory: ServiceFactory subclass
        """
        cls._factories[service_type.lower()] = factory
        logger.debug(f"Registered service factory: {service_type}")

    @classmethod
    def get_factory(cls, service_type: str) -> Type[ServiceFactory]:
        """
        Get factory for service type.

        Args:
            service_type: Service type name

        Returns:
            ServiceFactory class

        Raises:
            ValueError if service type not registered
        """
        key = service_type.lower()
        if key not in cls._factories:
            available = list(cls._factories.keys())
            raise ValueError(
                f"Unknown service type: {service_type}. "
                f"Available: {available}. "
                f"Register new types with ServiceRegistry.register()"
            )
        return cls._factories[key]

    @classmethod
    def list_service_types(cls) -> List[str]:
        """List all registered service types"""
        return list(cls._factories.keys())

    @classmethod
    def is_registered(cls, service_type: str) -> bool:
        """Check if service type is registered"""
        return service_type.lower() in cls._factories


# =============================================================================
# REGISTER BUILT-IN FACTORIES
# =============================================================================

# Object Detection
ServiceRegistry.register("object_detection", ObjectDetectionFactory)
ServiceRegistry.register("od", ObjectDetectionFactory)

# Camera View
ServiceRegistry.register("camera_view", CameraViewFactory)

# HLS Metadata
ServiceRegistry.register("hls_metadata", HLSMetadataFactory)

# Replay Detection (placeholder)
ServiceRegistry.register("replay_detection", ReplayDetectionFactory)

# Transcription (placeholder)
ServiceRegistry.register("transcription", TranscriptionFactory)

# Custom
ServiceRegistry.register("custom", CustomFactory)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_service(
    service_type: str,
    match_id: str,
    stream_url: str,
    service_config: "ServiceConfig",
    game_config: "GameConfig",
    **kwargs
) -> BaseService:
    """
    Create a service by type.

    Args:
        service_type: Service type (object_detection, camera_view, etc.)
        match_id: Match identifier
        stream_url: Stream URL
        service_config: ServiceConfig from schemas_v2
        game_config: Full GameConfig
        **kwargs: Additional parameters

    Returns:
        Configured service instance
    """
    factory_class = ServiceRegistry.get_factory(service_type)
    return factory_class.create(
        match_id=match_id,
        stream_url=stream_url,
        service_config=service_config,
        game_config=game_config,
        **kwargs
    )


def create_service_from_config(
    game_config: "GameConfig",
    service_id: str,
    match_id: str,
    stream_url: str,
    **kwargs
) -> BaseService:
    """
    Create a service from GameConfig by service ID.

    Args:
        game_config: GameConfig from config/schemas_v2.py
        service_id: Service ID to create
        match_id: Match identifier
        stream_url: Stream URL
        **kwargs: Additional parameters (start_frame, start_segment, local_output_dir)

    Returns:
        Configured service instance
    """
    service_config = game_config.get_service(service_id)
    if not service_config:
        available = [s.id for s in game_config.services]
        raise ValueError(
            f"Service not found: {service_id}. "
            f"Available services: {available}"
        )

    if not service_config.enabled:
        raise ValueError(f"Service {service_id} is disabled")

    return create_service(
        service_type=service_config.type,
        match_id=match_id,
        stream_url=stream_url,
        service_config=service_config,
        game_config=game_config,
        **kwargs
    )


def create_all_services(
    game_config: "GameConfig",
    match_id: str,
    stream_url: str,
    only_enabled: bool = True,
    **kwargs
) -> Dict[str, BaseService]:
    """
    Create all services defined in a GameConfig.

    Args:
        game_config: GameConfig
        match_id: Match identifier
        stream_url: Stream URL
        only_enabled: Only create enabled services (default True)
        **kwargs: Additional parameters

    Returns:
        Dict mapping service_id to service instance
    """
    services = {}

    service_configs = game_config.get_enabled_services() if only_enabled else game_config.services

    for service_config in service_configs:
        try:
            logger.info(f"Creating service: {service_config.id} ({service_config.type})")
            services[service_config.id] = create_service(
                service_type=service_config.type,
                match_id=match_id,
                stream_url=stream_url,
                service_config=service_config,
                game_config=game_config,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to create service {service_config.id}: {e}")
            raise

    return services


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "ServiceFactory",
    "ServiceRegistry",
    # Built-in factories
    "ObjectDetectionFactory",
    "CameraViewFactory",
    "HLSMetadataFactory",
    "ReplayDetectionFactory",
    "TranscriptionFactory",
    "CustomFactory",
    # Functions
    "create_service",
    "create_service_from_config",
    "create_all_services",
]
