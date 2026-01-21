"""
Service Registry

Central registry for all inference services.
Services self-register on import, eliminating if/elif chains in main.py.

This follows the Registry Pattern:
- Adding a new service = registration, not code modification
- Service selection is data-driven, not conditional logic

Usage:
    # Services self-register on import
    from services import od_service  # Registers ODService
    from services import pose_service  # Registers PoseService

    # Create service by type
    service = ServiceRegistry.create(
        service_type=ServiceType.OBJECT_DETECTION,
        match_id="match_123",
        source_url="/path/to/video.mp4",
        ...
    )
    service.run()
"""

import logging
from typing import Dict, Type, Callable, Any, Optional, List

from config.schemas import ServiceType
from services.base_service import BaseService, ServiceConfig

logger = logging.getLogger(__name__)


# Type alias for config builder function
ConfigBuilder = Callable[..., ServiceConfig]


class ServiceRegistry:
    """
    Registry for inference service types.

    Services register themselves with:
    - service_type: The ServiceType enum value
    - service_class: The service implementation class
    - config_builder: Function to build ServiceConfig from kwargs

    Why this exists:
        Before: if/elif chains in main.py for each service type
        After: Services self-register, main.py just calls ServiceRegistry.create()

    Benefits:
        - Add new service = add registration, no main.py changes
        - Service knowledge stays with service module
        - Easy to list available services
        - Testable in isolation
    """

    _services: Dict[ServiceType, Type[BaseService]] = {}
    _config_builders: Dict[ServiceType, ConfigBuilder] = {}
    _initialized: bool = False

    @classmethod
    def register(
        cls,
        service_type: ServiceType,
        service_class: Type[BaseService],
        config_builder: ConfigBuilder
    ):
        """
        Register a service type.

        Called by service modules during import to self-register.

        Args:
            service_type: The ServiceType enum value
            service_class: The service implementation class
            config_builder: Function that builds config from kwargs
        """
        if service_type in cls._services:
            logger.warning(f"Overwriting existing registration for {service_type}")

        cls._services[service_type] = service_class
        cls._config_builders[service_type] = config_builder
        logger.debug(f"Registered service: {service_type.value} -> {service_class.__name__}")

    @classmethod
    def create(cls, service_type: ServiceType, **kwargs) -> BaseService:
        """
        Create a service instance.

        Args:
            service_type: Type of service to create
            **kwargs: Arguments passed to config builder

        Returns:
            Configured service instance ready to run

        Raises:
            ValueError: If service type is not registered
        """
        cls._ensure_initialized()

        if service_type not in cls._services:
            available = [st.value for st in cls._services.keys()]
            raise ValueError(
                f"Unknown service type: {service_type}. "
                f"Available: {available}"
            )

        config_builder = cls._config_builders[service_type]
        service_class = cls._services[service_type]

        # Build config using the registered builder
        config = config_builder(**kwargs)

        # Create and return service instance
        logger.info(f"Creating service: {service_type.value}")
        return service_class(config)

    @classmethod
    def get_service_class(cls, service_type: ServiceType) -> Optional[Type[BaseService]]:
        """Get the service class for a type (without creating instance)."""
        cls._ensure_initialized()
        return cls._services.get(service_type)

    @classmethod
    def get_config_builder(cls, service_type: ServiceType) -> Optional[ConfigBuilder]:
        """Get the config builder for a type."""
        cls._ensure_initialized()
        return cls._config_builders.get(service_type)

    @classmethod
    def is_registered(cls, service_type: ServiceType) -> bool:
        """Check if a service type is registered."""
        return service_type in cls._services

    @classmethod
    def available_types(cls) -> List[ServiceType]:
        """Get list of all registered service types."""
        cls._ensure_initialized()
        return list(cls._services.keys())

    @classmethod
    def available_type_names(cls) -> List[str]:
        """Get list of all registered service type names (strings)."""
        cls._ensure_initialized()
        return [st.value for st in cls._services.keys()]

    @classmethod
    def clear(cls):
        """Clear all registrations (mainly for testing)."""
        cls._services.clear()
        cls._config_builders.clear()
        cls._initialized = False

    @classmethod
    def _ensure_initialized(cls):
        """
        Ensure services are registered by importing service modules.

        This is called lazily to avoid circular imports.
        """
        if cls._initialized:
            return

        # Import service modules to trigger self-registration
        # Each module registers itself when imported
        try:
            from services import od_service  # noqa: F401
        except ImportError as e:
            logger.debug(f"Could not import od_service: {e}")

        try:
            from services import pose_service  # noqa: F401
        except ImportError as e:
            logger.debug(f"Could not import pose_service: {e}")

        cls._initialized = True
        logger.debug(f"ServiceRegistry initialized with {len(cls._services)} services")


# =============================================================================
# HELPER FOR SERVICE MODULES
# =============================================================================

def register_service(
    service_type: ServiceType,
    service_class: Type[BaseService],
    config_builder: ConfigBuilder
):
    """
    Convenience function for registering a service.

    Usage in service module:
        from services.registry import register_service

        register_service(
            ServiceType.OBJECT_DETECTION,
            ODService,
            build_od_config
        )
    """
    ServiceRegistry.register(service_type, service_class, config_builder)
