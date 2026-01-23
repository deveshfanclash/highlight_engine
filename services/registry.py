"""
Service Registry

Simple registry for inference services.
Services register themselves on import via register().
"""

import logging
from typing import Dict, Type, Callable, Tuple, List

from config.schemas import ServiceType
from services.base_service import BaseService, ServiceConfig

logger = logging.getLogger(__name__)

# Type alias for config builder function
ConfigBuilder = Callable[..., ServiceConfig]

# Single registry: ServiceType -> (ServiceClass, ConfigBuilder)
_registry: Dict[ServiceType, Tuple[Type[BaseService], ConfigBuilder]] = {}


def register(
    service_type: ServiceType,
    service_class: Type[BaseService],
    config_builder: ConfigBuilder
):
    """
    Register a service type.

    Called by service modules during import to self-register.
    """
    if service_type in _registry:
        logger.warning(f"Overwriting existing registration for {service_type}")

    _registry[service_type] = (service_class, config_builder)
    logger.debug(f"Registered service: {service_type.value} -> {service_class.__name__}")


def create(service_type: ServiceType, **kwargs) -> BaseService:
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
    if service_type not in _registry:
        available = [st.value for st in _registry.keys()]
        raise ValueError(
            f"Unknown service type: {service_type}. "
            f"Available: {available}"
        )

    service_class, config_builder = _registry[service_type]
    config = config_builder(**kwargs)

    logger.info(f"Creating service: {service_type.value}")
    return service_class(config)


def available_types() -> List[ServiceType]:
    """Get list of all registered service types."""
    return list(_registry.keys())
