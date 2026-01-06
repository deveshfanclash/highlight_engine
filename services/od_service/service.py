"""
Object Detection Service

Concrete service implementation for running object detection models.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from config.schemas import InputType
from services.base_service import BaseService, ServiceConfig
from input_handlers import FrameInputPacket
from core.utils import ensure_model_available
from models.yolo_model import YOLOModel, create_yolo_model

logger = logging.getLogger(__name__)


@dataclass
class ODServiceConfig(ServiceConfig):
    """
    Configuration for Object Detection Service.

    Extends base ServiceConfig with model-specific settings.
    """
    # Model settings
    model_id: str = ""
    model_url: str = ""  # URL to download model from
    model_path: str = ""  # Local path (will be set after download)
    model_architecture: str = "yolov8"  # yolov8, yolov11, yolov12, custom

    # Inference settings
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    batch_size: int = 1
    half_precision: bool = False

    # Class filtering and mapping
    classes_to_predict: List[int] = field(default_factory=list)  # Model's native IDs
    class_mapping: Dict[int, str] = field(default_factory=dict)  # Model ID -> Universal name

    @classmethod
    def from_game_config(
        cls,
        match_id: str,
        input_source: str,
        model_config: Dict[str, Any],
        service_config: Dict[str, Any],
        inference_settings: Dict[str, Any],
        input_type: InputType = InputType.HLS
    ) -> "ODServiceConfig":
        """
        Create ODServiceConfig from GameConfig components.

        Args:
            match_id: Match identifier
            input_source: Stream URL or file path
            model_config: Model configuration from GameConfig
            service_config: Service configuration from GameConfig
            inference_settings: Global inference settings from GameConfig
            input_type: Type of input source
        """
        # Build class mapping from list format to dict
        class_mapping = {}
        for cm in model_config.get("class_mapping", []):
            if isinstance(cm, dict):
                class_mapping[cm["model_class_id"]] = cm["universal_class_name"]

        model_params = model_config.get("params", {})

        return cls(
            match_id=match_id,
            service_id=f"od_{model_config.get('model_id', 'unknown')}",
            input_source=input_source,
            input_type=input_type,
            target_width=inference_settings.get("processing_resolution", [1280, 720])[0],
            target_height=inference_settings.get("processing_resolution", [1280, 720])[1],
            frame_skip=inference_settings.get("frame_skip", 1),
            device=service_config.get("device", "cuda:0"),
            model_id=model_config.get("model_id", ""),
            model_url=model_config.get("model_url", ""),
            model_architecture=model_config.get("model_architecture", "yolov8"),
            confidence_threshold=model_params.get("confidence_threshold", 0.5),
            iou_threshold=model_params.get("iou_threshold", 0.45),
            max_detections=model_params.get("max_detections", 100),
            batch_size=model_params.get("batch_size", 1),
            half_precision=model_params.get("half_precision", False),
            classes_to_predict=model_config.get("classes_to_predict", []),
            class_mapping=class_mapping,
        )


class ODService(BaseService):
    """
    Object Detection Service.

    Runs YOLO (or other) object detection models on video frames.
    Each instance handles a single model - for multi-model games,
    spawn multiple ODService instances.

    Usage:
        config = ODServiceConfig(
            match_id="match_123",
            service_id="od_football_v2",
            stream_url="https://example.com/stream.m3u8",
            model_id="football_od_v2",
            model_url="s3://models/football/latest.pt",
            device="cuda:0",
            class_mapping={0: "PERSON", 1: "BALL"},
        )
        service = ODService(config)
        service.run()  # Blocking - processes until stream ends
    """

    def __init__(self, config: ODServiceConfig):
        super().__init__(config)
        self.od_config = config
        self._model: Optional[YOLOModel] = None

    def initialize(self) -> bool:
        """
        Initialize OD service - download and load model.

        Returns:
            True if initialization successful
        """
        try:
            # Determine local model path
            if self.od_config.model_path:
                local_path = self.od_config.model_path
            else:
                # Default: /tmp/models/{model_id}/latest.pt
                model_dir = os.path.join("/tmp", "models", self.od_config.model_id)
                os.makedirs(model_dir, exist_ok=True)
                local_path = os.path.join(model_dir, "latest.pt")

            # Ensure model is available (download if needed)
            if self.od_config.model_url and not os.path.exists(local_path):
                logger.info(f"Downloading model from {self.od_config.model_url}")
                local_path = ensure_model_available(
                    self.od_config.model_url,
                    local_path
                )

            # Load model
            logger.info(f"Loading model from {local_path}")
            self._model = create_yolo_model(
                model_id=self.od_config.model_id,
                model_path=local_path,
                device=self.od_config.device,
                class_mapping=self.od_config.class_mapping,
                half_precision=self.od_config.half_precision,
                warmup=True
            )

            logger.info(f"OD Service initialized: model={self.od_config.model_id}, device={self.od_config.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OD service: {e}")
            return False

    def process_frame(self, frame_packet: FrameInputPacket) -> Optional[Dict[str, Any]]:
        """
        Process a single frame - run object detection.

        Args:
            frame_packet: Frame data and metadata from input handler

        Returns:
            Dict with detections for DB storage
        """
        try:
            # Run inference
            outputs = self._model.predict(
                [frame_packet.frame],
                confidence_threshold=self.od_config.confidence_threshold,
                classes=self.od_config.classes_to_predict if self.od_config.classes_to_predict else None,
                iou_threshold=self.od_config.iou_threshold,
                max_detections=self.od_config.max_detections,
            )

            if not outputs:
                return None

            output = outputs[0]

            # Convert detections to dict format
            detections = output.to_dict_list()

            return {
                "model_id": self.od_config.model_id,
                "detections": detections,
            }

        except Exception as e:
            logger.error(f"Error processing frame {frame_packet.sequence_number}: {e}")
            return None

    def cleanup(self):
        """Clean up model resources"""
        if self._model:
            # YOLO models don't need explicit cleanup, but future models might
            self._model = None
        logger.info("OD Service cleaned up")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for running OD service standalone"""
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Object Detection Service")
    parser.add_argument("--match-id", required=True, help="Match identifier")
    parser.add_argument("--stream-url", required=True, help="Stream URL or file path")
    parser.add_argument("--input-type", default="hls", choices=["hls", "mp4", "file"], help="Input type")
    parser.add_argument("--model-id", required=True, help="Model identifier")
    parser.add_argument("--model-url", help="URL to download model from")
    parser.add_argument("--model-path", help="Local path to model file")
    parser.add_argument("--device", default="cuda:0", help="Device (cpu, cuda:0, etc.)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--class-mapping", type=str, help="JSON string of class mapping")
    parser.add_argument("--classes", type=str, help="Comma-separated class IDs to predict")
    parser.add_argument("--width", type=int, help="Processing width")
    parser.add_argument("--height", type=int, help="Processing height")
    parser.add_argument("--db-table", default="inference_results", help="DynamoDB table name")
    parser.add_argument("--local-output", help="Local output directory (for testing without DynamoDB)")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame number to start from (for resume)")
    parser.add_argument("--start-segment", type=int, default=1, help="Segment number to start from (for HLS resume)")

    args = parser.parse_args()

    # Map input type string to enum
    input_type_map = {"hls": InputType.HLS, "mp4": InputType.MP4, "file": InputType.FILE}
    input_type = input_type_map.get(args.input_type, InputType.HLS)

    # Parse class mapping
    class_mapping = {}
    if args.class_mapping:
        class_mapping = json.loads(args.class_mapping)
        # Convert string keys to int
        class_mapping = {int(k): v for k, v in class_mapping.items()}

    # Parse classes to predict
    classes_to_predict = []
    if args.classes:
        classes_to_predict = [int(c.strip()) for c in args.classes.split(",")]

    # Create config
    config = ODServiceConfig(
        match_id=args.match_id,
        service_id=f"od_{args.model_id}",
        input_source=args.stream_url,
        input_type=input_type,
        model_id=args.model_id,
        model_url=args.model_url or "",
        model_path=args.model_path or "",
        device=args.device,
        confidence_threshold=args.confidence,
        class_mapping=class_mapping,
        classes_to_predict=classes_to_predict,
        target_width=args.width,
        target_height=args.height,
        db_table_name=args.db_table,
        local_output_dir=args.local_output,
        start_frame=args.start_frame,
        start_segment=args.start_segment,
    )

    # Run service
    service = ODService(config)
    service.run()


if __name__ == "__main__":
    main()
