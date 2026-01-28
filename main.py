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

    # Production or development mode (requires AWS credentials via env vars)
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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config.loader import load_service_configs
from config.environment import get_infra_config
from core.resume import ResumeMode, get_resume_position
from services import registry

# Import services to trigger registration
from services import od_service  # noqa: F401

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference System Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--game-config", "-g", required=True, help="Path to game config YAML")
    parser.add_argument("--match-id", "-m", required=True, help="Match identifier")
    parser.add_argument("--source", "-s", required=True, help="Stream URL or file path")

    # Optional arguments
    parser.add_argument("--resume", "-r", choices=["start", "current", "latest"], default="start")
    parser.add_argument("--service-type", help="Specific service type to run (default: first enabled)")
    parser.add_argument("--local", "-l", action="store_true", help="Run in local mode")
    parser.add_argument("--base-path", "-b", help="Base path for local mode output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # Video output
    parser.add_argument("--video-output", action="store_true", help="Enable annotated video output")
    parser.add_argument("--video-output-path", help="Path to output video file")

    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    args = parse_args()
    setup_logging(args.verbose)

    # Initialize infrastructure config
    infra = get_infra_config(local_mode=args.local, base_path=args.base_path)
    base_path = args.base_path or str(infra.base_path)

    if args.local:
        logger.info("Running in LOCAL MODE - no AWS credentials required")

    # Load resolved service configs
    logger.info(f"Loading config: {args.game_config}")
    service_configs = load_service_configs(args.game_config)

    if not service_configs:
        logger.error("No enabled services found in config")
        sys.exit(1)

    # Select service to run
    if args.service_type:
        target = next(
            (cfg for cfg in service_configs if cfg.service_type.value == args.service_type),
            None
        )
        if not target:
            logger.error(f"Service type '{args.service_type}' not found or not enabled")
            sys.exit(1)
    else:
        target = service_configs[0]

    logger.info(f"Running service: {target.service_id} on {target.device}")
    logger.info(f"Model: {target.model_config.get('model_id')}")

    # Calculate resume position
    resume_mode = ResumeMode(args.resume)
    resume_position = get_resume_position(
        mode=resume_mode,
        match_id=args.match_id,
        service_id=target.service_id,
        source_url=args.source,
        table_name=target.output_settings.get("db_table_name", "inference_results"),
        local_mode=args.local,
    )
    logger.info(f"Resume: frame={resume_position.frame_number}, segment={resume_position.segment_number}")

    # Build service config dict (merge output settings into service config)
    service_config_dict = {
        "device": target.device,
        **target.output_settings,
        **target.extra,
    }

    # Create service
    try:
        service = registry.create(
            service_type=target.service_type,
            match_id=args.match_id,
            source_url=args.source,
            model_config=target.model_config,
            service_config=service_config_dict,
            inference_settings=target.inference_settings,
            start_frame=max(0, resume_position.frame_number),
            start_segment=max(0, resume_position.segment_number),
            local_mode=args.local,
            output_dir=base_path,
            video_output_enabled=args.video_output,
            video_output_path=args.video_output_path,
        )
    except ValueError as e:
        logger.error(f"Failed to create service: {e}")
        sys.exit(1)

    # Run
    logger.info("Starting service...")
    try:
        service.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    logger.info("Service completed")


if __name__ == "__main__":
    main()
