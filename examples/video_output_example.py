#!/usr/bin/env python3
"""
Video Output Example

Demonstrates how to use VideoOutputHandler for local testing.
Processes a video file and writes annotated output.

Usage:
    # Process video with annotated output
    python examples/video_output_example.py \
        --source /path/to/video.mp4 \
        --output output/annotated.mp4 \
        --video-output

    # Process without video output (inference only)
    python examples/video_output_example.py \
        --source /path/to/video.mp4

    # With custom settings
    python examples/video_output_example.py \
        --source /path/to/video.mp4 \
        --output output/result.mp4 \
        --video-output \
        --confidence 0.5 \
        --show-labels \
        --no-show-confidence
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from core.video_output import create_video_output, VideoOutputHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_mock_detections(frame_num: int, width: int, height: int) -> dict:
    """
    Create mock detection results for demonstration.

    In real usage, this would come from ODService.process_frame()
    """
    # Simulate a moving detection
    x_offset = (frame_num % 100) / 100.0 * 0.5

    return {
        "model_id": "demo_model",
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "bbox": {
                    "x1": 0.1 + x_offset,
                    "y1": 0.2,
                    "x2": 0.25 + x_offset,
                    "y2": 0.8,
                },
            },
            {
                "class_id": 1,
                "class_name": "ball",
                "confidence": 0.88,
                "bbox": {
                    "x1": 0.5 + x_offset * 0.3,
                    "y1": 0.6,
                    "x2": 0.55 + x_offset * 0.3,
                    "y2": 0.65,
                },
            },
        ],
        "detection_count": 2,
    }


def process_video_with_mock_inference(
    source_path: str,
    output_path: str,
    enable_video_output: bool = True,
    show_labels: bool = True,
    show_confidence: bool = True,
    box_thickness: int = 2,
) -> None:
    """
    Process a video file with mock inference and optional video output.

    This demonstrates the pattern for integrating VideoOutputHandler
    with a video processing loop.
    """
    # Open source video
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {source_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Source: {source_path}")
    logger.info(f"Resolution: {width}x{height} @ {fps}fps")
    logger.info(f"Total frames: {total_frames}")

    # Initialize video output handler (only if enabled)
    video_handler: VideoOutputHandler = None
    if enable_video_output:
        video_handler = create_video_output(
            output_path=output_path,
            fps=fps,
            annotate=True,
            show_labels=show_labels,
            show_confidence=show_confidence,
            box_thickness=box_thickness,
        )
        logger.info(f"Video output enabled: {output_path}")

    # Process frames
    frame_num = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # === INFERENCE WOULD GO HERE ===
            # In real usage:
            #   result = service.process_frame(frame_packet)
            # For demo, we use mock detections:
            result = create_mock_detections(frame_num, width, height)

            # Write to video output if enabled
            if video_handler:
                video_handler.write_frame(frame, result)

            frame_num += 1

            # Progress logging
            if frame_num % 100 == 0:
                logger.info(f"Processed {frame_num}/{total_frames} frames")

    finally:
        cap.release()

        if video_handler:
            video_handler.close()

    logger.info(f"Processing complete: {frame_num} frames")


def process_with_real_inference(
    source_path: str,
    output_path: str,
    game_config_path: str,
    enable_video_output: bool = True,
) -> None:
    """
    Example of integrating video output with real ODService.

    This shows the pattern for manual integration with services.
    """
    # This is a template - uncomment and modify for real usage:
    #
    # from config.loader import ConfigLoader
    # from services.od_service import ODService, build_od_config
    # from core.source_router import SourceRouter
    # from core.video_output import create_video_output
    #
    # # Load config
    # game_template, model_registry = ConfigLoader.load_from_yaml(game_config_path)
    #
    # # Get service and model config
    # service_template = game_template.get_enabled_services()[0]
    # model = model_registry.get_model(service_template.model_id)
    #
    # # Build service config
    # config = build_od_config(
    #     match_id="local_test",
    #     source_url=source_path,
    #     model_config=model.model_dump(),
    #     service_config=service_template.model_dump(),
    #     inference_settings=game_template.resolve_inference_settings(service_template),
    #     local_mode=True,
    # )
    #
    # # Create service
    # service = ODService(config)
    # service.setup()
    #
    # # Create video output handler
    # video_handler = None
    # if enable_video_output:
    #     video_handler = create_video_output(
    #         output_path=output_path,
    #         fps=service.fps or 30.0,
    #         annotate=True,
    #     )
    #
    # # Process frames
    # try:
    #     for frame_packet in service._input_handler.iterate():
    #         # Run inference
    #         result = service.process_frame(frame_packet)
    #
    #         # Write annotated frame
    #         if video_handler and result:
    #             video_handler.write_frame(frame_packet.frame, result)
    #
    # finally:
    #     if video_handler:
    #         video_handler.close()
    #     service.shutdown()

    logger.info("Real inference integration - see code comments for implementation")
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Video Output Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Source video path",
    )
    parser.add_argument(
        "--output", "-o",
        default="output/annotated.mp4",
        help="Output video path (default: output/annotated.mp4)",
    )
    parser.add_argument(
        "--video-output",
        action="store_true",
        help="Enable video output (flag for local testing)",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        default=True,
        help="Show class labels (default: True)",
    )
    parser.add_argument(
        "--no-show-labels",
        action="store_false",
        dest="show_labels",
        help="Hide class labels",
    )
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        default=True,
        help="Show confidence scores (default: True)",
    )
    parser.add_argument(
        "--no-show-confidence",
        action="store_false",
        dest="show_confidence",
        help="Hide confidence scores",
    )
    parser.add_argument(
        "--box-thickness",
        type=int,
        default=2,
        help="Bounding box thickness (default: 2)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process video
    process_video_with_mock_inference(
        source_path=args.source,
        output_path=str(output_path),
        enable_video_output=args.video_output,
        show_labels=args.show_labels,
        show_confidence=args.show_confidence,
        box_thickness=args.box_thickness,
    )


if __name__ == "__main__":
    main()
