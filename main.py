#!/usr/bin/env python3
"""
Inference System Entry Point

Usage:
    # Local mode (no AWS credentials needed)
    python main.py \
      --game-config config/games/football_v1.yaml \
      --match-id test_123 \
      --source /path/to/video.mp4 \
      --local

    # Production mode (requires AWS credentials via env vars)
    python main.py \
      --game-config config/games/football_v1.yaml \
      --match-id match_12345 \
      --source "https://cdn.example.com/stream.m3u8" \
      --resume current

Resume Modes:
    - start:   Fresh start from frame 0
    - current: Resume from last written frame (queries DB, skipped in local mode)
    - latest:  Start from current stream position (live edge)
"""

import argparse
import logging
import sys
from typing import Optional

from config.loader import ConfigLoader
from config.environment import get_infra_config
from config.schemas import ServiceType
from core.resume import ResumeMode, get_resume_position
from services.registry import ServiceRegistry

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging for the application"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

#Load dotenv from .env file if not from orchestrator or shell set with env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def run_service(
    game_config_path: str,
    match_id: str,
    source_url: str,
    resume_mode: ResumeMode,
    service_type: Optional[str] = None,
    local_mode: bool = False,
    base_path: Optional[str] = None,
):
    """
    Run an inference service based on game configuration.

    Args:
        game_config_path: Path to game config YAML file
        match_id: Match identifier
        source_url: Stream URL or file path
        resume_mode: Resume mode (start, current, latest)
        service_type: Specific service type to run (optional): Current system supports only one service at a time
        local_mode: If True, run without infrastructure dependencies
        base_path: Base path directory for local mode
    """
    # Initialize infrastructure config
    infra = get_infra_config(local_mode=local_mode, base_path=base_path)

    if local_mode:
        logger.info("Running in LOCAL MODE - no AWS credentials required")
        base_path = base_path or str(infra.base_path)

    # Load game configuration
    logger.info(f"Loading game config from: {game_config_path}")
    game_template, model_registry = ConfigLoader.load_from_yaml(game_config_path)

    logger.info(f"Game: {game_template.game_name} (ID: {game_template.game_id})")
    logger.info(f"Models loaded: {len(model_registry.models)}")
    logger.info(f"Services configured: {len(game_template.services)}")

    # Get enabled services
    enabled_services = game_template.get_enabled_services()
    if not enabled_services:
        logger.error("No enabled services found in game config")
        return

    # Select service to run start------------------------------
    target_service = None
    if service_type:
        for service in enabled_services:
            if service.service_type.value == service_type:
                target_service = service
                break
        if not target_service:
            logger.error(f"Service type '{service_type}' not found or not enabled")
            return
    else:
        target_service = enabled_services[0]

    logger.info(f"Running service: {target_service.service_type.value}")
    #----Code to add service override for a single type-----

    # Get model for this service (direct lookup by model_id)
    model_id = getattr(target_service, 'model_id', None)
    if not model_id:
        logger.error(f"Service {target_service.service_type.value} has no model_id configured")
        return

    model = model_registry.get_model(model_id)
    if not model:
        logger.error(f"Model '{model_id}' not found in registry")
        return

    model_config = model.model_dump()
    logger.info(f"Using model: {model_config['model_id']}")

    # Resolve settings: defaults + service-specific overrides
    inference_settings = game_template.resolve_inference_settings(target_service)
    output_settings = game_template.resolve_output_settings(target_service)

    # Build service config dict with resolved settings
    service_config_dict = target_service.model_dump()
    service_config_dict.update(output_settings)

    # Log effective settings
    if inference_settings["num_workers"] > 1:
        logger.info(f"Multi-worker mode: {inference_settings['num_workers']} workers")

    # Calculate resume position (using resolved db_table_name)
    service_id = f"{target_service.service_type.value}_{model_config['model_id']}"
    db_table_name = output_settings.get("db_table_name", "inference_results")

    resume_position = get_resume_position(
        mode=resume_mode,
        match_id=match_id,
        service_id=service_id,
        source_url=source_url,
        table_name=db_table_name,
        local_mode=local_mode,
    )

    logger.info(
        f"Resume position: frame={resume_position.frame_number}, "
        f"segment={resume_position.segment_number}"
    )

    effective_device = service_config_dict.get("device", "cuda:0")
    logger.info(f"Device: {effective_device}")

    try:
        service = ServiceRegistry.create(
            service_type=target_service.service_type,
            match_id=match_id,
            source_url=source_url,
            model_config=model_config,
            service_config=service_config_dict,
            inference_settings=inference_settings,
            start_frame=max(0, resume_position.frame_number),
            start_segment=max(0, resume_position.segment_number),
            local_mode=local_mode,
            output_dir=base_path,
        )
    except ValueError as e:
        logger.error(f"Failed to create service: {e}")
        return

    # Run the service (blocking)
    logger.info("Starting service...")
    service.run()
    logger.info("Service completed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Inference System Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""CLI Overrides: 
        --device overrides the device setting from service config.
        If not provided, the device value from the YAML config is used.
        """
    )

    parser.add_argument(
        "--game-config", "-g",
        required=True,
        help="Path to game config YAML file"
    )
    parser.add_argument(
        "--match-id", "-m",
        required=True,
        help="Match identifier"
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Stream URL or file path"
    )
    parser.add_argument(
        "--resume", "-r",
        choices=["start", "current", "latest"],
        default="start",
        help="Resume mode (default: start)"
    )
    parser.add_argument(
        "--service-type",
        help="Specific service type to run (default: first enabled)"
    )
    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Run in local mode (no AWS credentials required, outputs to files)"
    )
    parser.add_argument(
        "--base-path", "-b",
        help="Base path directory for local mode (default: ./output)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Map resume string to enum
    resume_mode = ResumeMode(args.resume)

    # Run the service
    try:
        run_service(
            game_config_path=args.game_config,
            match_id=args.match_id,
            source_url=args.source,
            resume_mode=resume_mode,
            service_type=args.service_type,
            local_mode=args.local,
            base_path=args.base_path,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running service: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
