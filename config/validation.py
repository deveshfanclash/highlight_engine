"""
Configuration Validation

Validates GameConfig against model and service registries.
Ensures configs are valid before runtime.

Usage:
    from config.validation import validate_config, ConfigValidationError

    try:
        validate_config(game_config)
    except ConfigValidationError as e:
        print(f"Config validation failed: {e}")
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""

    def __init__(self, message: str, errors: List[str] = None):
        self.errors = errors or [message]
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if len(self.errors) == 1:
            return self.errors[0]
        return f"Multiple validation errors:\n" + "\n".join(f"  - {e}" for e in self.errors)


@dataclass
class ValidationResult:
    """Result of config validation"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def merge(self, other: "ValidationResult"):
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False


def validate_config(
    game_config: "GameConfig",
    check_registry: bool = True,
    strict: bool = False
) -> ValidationResult:
    """
    Validate a GameConfig.

    Performs:
    1. Schema validation (already done by Pydantic)
    2. Cross-reference validation (services reference valid models)
    3. Registry validation (architectures and service types are registered)
    4. Device validation (CUDA devices exist if specified)

    Args:
        game_config: GameConfig to validate
        check_registry: Check against model/service registries (default True)
        strict: Treat warnings as errors (default False)

    Returns:
        ValidationResult with errors and warnings

    Raises:
        ConfigValidationError if validation fails and strict=True
    """
    result = ValidationResult(valid=True)

    # 1. Cross-reference validation
    result.merge(_validate_model_references(game_config))

    # 2. Registry validation
    if check_registry:
        result.merge(_validate_architectures(game_config))
        result.merge(_validate_service_types(game_config))

    # 3. Device validation
    result.merge(_validate_devices(game_config))

    # 4. Output validation
    result.merge(_validate_output(game_config))

    # Log results
    if result.warnings:
        for warning in result.warnings:
            logger.warning(f"Config warning: {warning}")

    if not result.valid:
        for error in result.errors:
            logger.error(f"Config error: {error}")

    # Strict mode: treat warnings as errors
    if strict and result.warnings:
        result.errors.extend(result.warnings)
        result.valid = False

    if not result.valid:
        raise ConfigValidationError("Config validation failed", result.errors)

    return result


def _validate_model_references(game_config: "GameConfig") -> ValidationResult:
    """Validate that all service model references exist"""
    result = ValidationResult(valid=True)

    model_ids = {m.id for m in game_config.models}

    for service in game_config.services:
        for model_id in service.models:
            if model_id not in model_ids:
                result.add_error(
                    f"Service '{service.id}' references unknown model '{model_id}'. "
                    f"Available models: {sorted(model_ids)}"
                )

    return result


def _validate_architectures(game_config: "GameConfig") -> ValidationResult:
    """Validate that all model architectures are registered"""
    result = ValidationResult(valid=True)

    try:
        from models.registry import ModelRegistry
    except ImportError:
        result.add_warning("ModelRegistry not available, skipping architecture validation")
        return result

    registered = set(ModelRegistry.list_architectures())

    for model in game_config.models:
        arch = model.architecture.lower()
        if arch not in registered:
            result.add_error(
                f"Model '{model.id}' uses unregistered architecture '{model.architecture}'. "
                f"Registered: {sorted(registered)}. "
                f"Register with ModelRegistry.register() or use architecture='custom'."
            )

    return result


def _validate_service_types(game_config: "GameConfig") -> ValidationResult:
    """Validate that all service types are registered"""
    result = ValidationResult(valid=True)

    try:
        from services.registry import ServiceRegistry
    except ImportError:
        result.add_warning("ServiceRegistry not available, skipping service type validation")
        return result

    registered = set(ServiceRegistry.list_service_types())

    for service in game_config.services:
        service_type = service.type.lower()
        if service_type not in registered:
            result.add_error(
                f"Service '{service.id}' uses unregistered type '{service.type}'. "
                f"Registered: {sorted(registered)}. "
                f"Register with ServiceRegistry.register() or use type='custom'."
            )

    return result


def _validate_devices(game_config: "GameConfig") -> ValidationResult:
    """Validate device specifications"""
    result = ValidationResult(valid=True)

    cuda_devices = set()

    for service in game_config.services:
        device = service.device

        # Collect CUDA devices for duplicate check
        if device.startswith("cuda:"):
            cuda_devices.add(device)

        # Validate CUDA device format
        if device.startswith("cuda:"):
            try:
                device_id = int(device.split(":")[1])
                if device_id < 0:
                    result.add_error(f"Service '{service.id}' has invalid device: {device}")
            except ValueError:
                result.add_error(f"Service '{service.id}' has invalid device format: {device}")

    # Check for CUDA availability (warning only)
    if cuda_devices:
        try:
            import torch
            if not torch.cuda.is_available():
                result.add_warning(
                    f"Services use CUDA devices {sorted(cuda_devices)} but CUDA is not available"
                )
            else:
                cuda_count = torch.cuda.device_count()
                for device in cuda_devices:
                    device_id = int(device.split(":")[1])
                    if device_id >= cuda_count:
                        result.add_warning(
                            f"Device {device} specified but only {cuda_count} CUDA devices available"
                        )
        except ImportError:
            result.add_warning("torch not available, cannot validate CUDA devices")

    return result


def _validate_output(game_config: "GameConfig") -> ValidationResult:
    """Validate output configuration"""
    result = ValidationResult(valid=True)

    output = game_config.output

    # Validate primary output
    primary_type = output.primary.get("type")
    if not primary_type:
        result.add_error("Output primary must specify 'type'")
    elif primary_type == "dynamodb":
        if not output.primary.get("table"):
            result.add_error("DynamoDB output must specify 'table'")
    elif primary_type == "s3":
        if not output.primary.get("bucket"):
            result.add_error("S3 output must specify 'bucket'")
    elif primary_type not in ("local", "kafka"):
        result.add_warning(f"Unknown output type: {primary_type}")

    # Validate secondary output if present
    if output.secondary:
        secondary_type = output.secondary.get("type")
        if secondary_type == "s3":
            if not output.secondary.get("bucket"):
                result.add_error("S3 secondary output must specify 'bucket'")

    return result


def validate_model_config(model_config: "ModelConfig") -> ValidationResult:
    """Validate a single ModelConfig"""
    result = ValidationResult(valid=True)

    # Check ID format
    if not model_config.id.replace("_", "").isalnum():
        result.add_error(f"Model ID must be alphanumeric with underscores: {model_config.id}")

    # Check weights
    if not model_config.weights:
        result.add_error(f"Model '{model_config.id}' missing weights path")

    # Check custom loader if architecture is 'custom'
    if model_config.architecture == "custom":
        if not model_config.custom_loader:
            result.add_error(
                f"Model '{model_config.id}' uses architecture='custom' but "
                f"custom_loader is not specified"
            )
        else:
            if not model_config.custom_loader.get("module"):
                result.add_error(f"Model '{model_config.id}' custom_loader missing 'module'")
            if not model_config.custom_loader.get("class_name"):
                result.add_error(f"Model '{model_config.id}' custom_loader missing 'class_name'")

    return result


def validate_service_config(
    service_config: "ServiceConfig",
    game_config: "GameConfig"
) -> ValidationResult:
    """Validate a single ServiceConfig in context of GameConfig"""
    result = ValidationResult(valid=True)

    # Check ID format
    if not service_config.id.replace("_", "").isalnum():
        result.add_error(f"Service ID must be alphanumeric with underscores: {service_config.id}")

    # Check model references for model-based services
    model_based_types = {"object_detection", "od", "segmentation", "transcription"}
    if service_config.type.lower() in model_based_types:
        if not service_config.models:
            result.add_warning(
                f"Service '{service_config.id}' is type '{service_config.type}' "
                f"but has no models specified"
            )

    # Check custom service params
    if service_config.type == "custom":
        params = service_config.params or {}
        if not params.get("module"):
            result.add_error(
                f"Service '{service_config.id}' uses type='custom' but "
                f"params.module is not specified"
            )
        if not params.get("class_name"):
            result.add_error(
                f"Service '{service_config.id}' uses type='custom' but "
                f"params.class_name is not specified"
            )

    return result


def quick_validate(config_path: str) -> bool:
    """
    Quick validation of a config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        True if valid, False otherwise
    """
    try:
        from config.loader_v2 import load_config
        game_config = load_config(config_path)
        result = validate_config(game_config, check_registry=True, strict=False)
        return result.valid
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for validating config files"""
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Validate game configuration")
    parser.add_argument("config_path", help="Path to config file")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument("--no-registry", action="store_true", help="Skip registry validation")

    args = parser.parse_args()

    try:
        from config.loader_v2 import load_config

        print(f"Loading config: {args.config_path}")
        game_config = load_config(args.config_path)

        print(f"Validating config for game: {game_config.name}")
        result = validate_config(
            game_config,
            check_registry=not args.no_registry,
            strict=args.strict
        )

        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  ! {w}")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for e in result.errors:
                print(f"  X {e}")
            sys.exit(1)
        else:
            print("\nValidation PASSED")
            print(f"  Models: {len(game_config.models)}")
            print(f"  Services: {len(game_config.services)} ({len(game_config.get_enabled_services())} enabled)")
            sys.exit(0)

    except ConfigValidationError as e:
        print(f"\nValidation FAILED:")
        for error in e.errors:
            print(f"  X {error}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
