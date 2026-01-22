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


def run_service(
    game_config_path: str,
    match_id: str,
    source_url: str,
    resume_mode: ResumeMode,
    service_type: Optional[str] = None,
    device: str = "cuda:0",
    local_mode: bool = False,
    output_dir: Optional[str] = None,
    num_workers: int = 1,
    queue_size: int = 0,
    buffer_mode: str = "drop_old",
):
    """
    Run an inference service based on game configuration.

    Args:
        game_config_path: Path to game config YAML file
        match_id: Match identifier
        source_url: Stream URL or file path
        resume_mode: Resume mode (start, current, latest)
        service_type: Specific service type to run (optional)
        device: Device to run on (cpu, cuda:0, etc.)
        local_mode: If True, run without infrastructure dependencies
        output_dir: Output directory for local mode
        num_workers: Number of worker processes for parallel inference
        queue_size: Queue size per worker (0 = auto)
        buffer_mode: Buffer mode for streams ('fifo' or 'drop_old')
    """
    # Initialize infrastructure config
    infra = get_infra_config(local_mode=local_mode, output_dir=output_dir)

    if local_mode:
        logger.info("Running in LOCAL MODE - no AWS credentials required")
        output_dir = output_dir or str(infra.output_path)

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

    # Select service to run
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

    # Get model for this service
    model_roles = getattr(target_service, 'model_roles', ['default'])
    model_config = None

    for role in model_roles:
        assignment = game_template.get_model_assignment(role)
        if assignment:
            model = model_registry.get_model(assignment.model_id)
            if model:
                model_config = model.model_dump()
                if assignment.params_override:
                    model_config["default_params"] = {
                        **model_config.get("default_params", {}),
                        **assignment.params_override.model_dump(exclude_none=True)
                    }
                break

    if not model_config:
        logger.error(f"No model found for service roles: {model_roles}")
        return

    logger.info(f"Using model: {model_config['model_id']}")

    # Calculate resume position
    service_id = f"{target_service.service_type.value}_{model_config['model_id']}"
    db_table_name = getattr(target_service, 'db_table_name', 'inference_results')

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

    # Create service using ServiceRegistry
    # The registry handles config building and service instantiation
    inference_settings = game_template.inference_settings.model_dump()

    # CLI overrides for multi-worker settings
    if num_workers > 1:
        inference_settings["num_workers"] = num_workers
        logger.info(f"Multi-worker mode enabled: {num_workers} workers")
    if queue_size > 0:
        inference_settings["worker_queue_size"] = queue_size
    if buffer_mode:
        inference_settings["buffer_mode"] = buffer_mode

    service_config_dict = target_service.model_dump()

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
            device=device,
            local_mode=local_mode,
            output_dir=output_dir,
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
        epilog="""
Examples:
  # Local mode (no AWS credentials needed)
  python main.py \\
    --game-config config/games/football_v1.yaml \\
    --match-id test_123 \\
    --source /path/to/video.mp4 \\
    --local

  # Production mode with DynamoDB output
  python main.py \\
    --game-config config/games/football_v1.yaml \\
    --match-id match_12345 \\
    --source "https://cdn.example.com/stream.m3u8" \\
    --resume current

Resume Modes:
  start   - Fresh start from frame 0
  current - Resume from last written frame (queries DB, skipped in local mode)
  latest  - Start from current stream position (live edge)

Multi-Worker Mode:
  # Run with 4 parallel worker processes
  python main.py \\
    --game-config config/games/football_v1.yaml \\
    --match-id match_12345 \\
    --source "https://cdn.example.com/stream.m3u8" \\
    --num-workers 4
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
        "--device", "-d",
        default="cuda:0",
        help="Device to run on (default: cuda:0)"
    )
    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Run in local mode (no AWS credentials required, outputs to files)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for local mode (default: ./output)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    # Multi-worker settings
    parser.add_argument(
        "--num-workers", "-w",
        type=int,
        default=1,
        help="Number of worker processes for parallel inference (default: 1)"
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=0,
        help="Queue size per worker (default: 0 = auto, num_workers * 4)"
    )
    parser.add_argument(
        "--buffer-mode",
        choices=["fifo", "drop_old"],
        default="drop_old",
        help="Buffer mode for streams (default: drop_old for real-time)"
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
            device=args.device,
            local_mode=args.local,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            queue_size=args.queue_size,
            buffer_mode=args.buffer_mode,
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
