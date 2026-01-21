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
from typing import Optional, Dict, Any

from config.loader import ConfigLoader
from config.environment import get_infra_config
from config.schemas import InputType, ServiceType
from core.resume import ResumeMode, get_resume_position
from services.od_service import ODService, ODServiceConfig
from services.pose_service import PoseService, PoseServiceConfig

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging for the application"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def get_input_type(source_url: str) -> InputType:
    """Determine input type from source URL"""
    source_lower = source_url.lower()

    if source_lower.endswith('.m3u8') or 'hls' in source_lower:
        return InputType.HLS
    elif source_lower.endswith('.mp4'):
        return InputType.MP4
    elif source_lower.startswith('/') or source_lower.startswith('file://'):
        return InputType.FILE
    elif source_lower.startswith('rtsp://'):
        return InputType.RTSP
    else:
        return InputType.HLS


def create_od_service_config(
    match_id: str,
    source_url: str,
    model_config: Dict[str, Any],
    service_config: Dict[str, Any],
    inference_settings: Dict[str, Any],
    start_frame: int = 0,
    start_segment: int = 0,
    device: str = "cuda:0",
    local_mode: bool = False,
    output_dir: Optional[str] = None,
) -> ODServiceConfig:
    """Create ODServiceConfig from game config components"""
    input_type = get_input_type(source_url)

    # Build class mapping from list format to dict
    class_mapping = {}
    for cm in model_config.get("default_class_mapping", []):
        if isinstance(cm, dict):
            class_mapping[cm["model_class_id"]] = cm["universal_class_name"]

    model_params = model_config.get("default_params", {})
    processing_resolution = inference_settings.get("processing_resolution", [1280, 720])

    return ODServiceConfig(
        match_id=match_id,
        service_id=f"od_{model_config.get('model_id', 'unknown')}",
        input_source=source_url,
        input_type=input_type,
        target_width=processing_resolution[0] if processing_resolution else None,
        target_height=processing_resolution[1] if len(processing_resolution) > 1 else None,
        frame_skip=inference_settings.get("frame_skip", 1),
        start_frame=start_frame,
        start_segment=start_segment,
        device=device,
        db_table_name=service_config.get("db_table_name", "inference_results"),
        db_batch_size=service_config.get("db_batch_size", 12),
        db_flush_interval_ms=service_config.get("db_flush_interval_ms", 250),
        local_mode=local_mode,
        local_output_dir=output_dir,
        model_id=model_config.get("model_id", ""),
        model_url=model_config.get("model_url", ""),
        model_architecture=model_config.get("model_architecture", "yolov8"),
        confidence_threshold=model_params.get("confidence_threshold", 0.5),
        iou_threshold=model_params.get("iou_threshold", 0.45),
        max_detections=model_params.get("max_detections", 100),
        half_precision=model_params.get("half_precision", False),
        classes_to_predict=model_config.get("classes_to_predict", []),
        class_mapping=class_mapping,
    )


def create_pose_service_config(
    match_id: str,
    source_url: str,
    model_config: Dict[str, Any],
    service_config: Dict[str, Any],
    inference_settings: Dict[str, Any],
    start_frame: int = 0,
    start_segment: int = 0,
    device: str = "cuda:0",
    local_mode: bool = False,
    output_dir: Optional[str] = None,
) -> PoseServiceConfig:
    """Create PoseServiceConfig from game config components"""
    input_type = get_input_type(source_url)
    model_params = model_config.get("default_params", {})
    processing_resolution = inference_settings.get("processing_resolution", [1280, 720])

    return PoseServiceConfig(
        match_id=match_id,
        service_id=f"pose_{model_config.get('model_id', 'unknown')}",
        input_source=source_url,
        input_type=input_type,
        target_width=processing_resolution[0] if processing_resolution else None,
        target_height=processing_resolution[1] if len(processing_resolution) > 1 else None,
        frame_skip=inference_settings.get("frame_skip", 1),
        start_frame=start_frame,
        start_segment=start_segment,
        device=device,
        db_table_name=service_config.get("db_table_name", "inference_results"),
        db_batch_size=service_config.get("db_batch_size", 12),
        db_flush_interval_ms=service_config.get("db_flush_interval_ms", 250),
        local_mode=local_mode,
        local_output_dir=output_dir,
        model_id=model_config.get("model_id", ""),
        model_url=model_config.get("model_url", ""),
        model_architecture=model_config.get("model_architecture", "yolov8-pose"),
        confidence_threshold=model_params.get("confidence_threshold", 0.5),
        iou_threshold=model_params.get("iou_threshold", 0.45),
        max_detections=model_params.get("max_detections", 100),
        half_precision=model_params.get("half_precision", False),
        keypoint_confidence_threshold=service_config.get("keypoint_confidence_threshold", 0.5),
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

    # Create service config
    inference_settings = game_template.inference_settings.model_dump()
    service_config_dict = target_service.model_dump()

    # Create and run service based on type
    if target_service.service_type == ServiceType.OBJECT_DETECTION:
        config = create_od_service_config(
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
        service = ODService(config)

    elif target_service.service_type == ServiceType.POSE_ESTIMATION:
        config = create_pose_service_config(
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
        service = PoseService(config)

    else:
        logger.error(f"Unsupported service type: {target_service.service_type}")
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
