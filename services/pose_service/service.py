"""
Pose Estimation Service

Concrete service implementation for running pose estimation models.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from config.schemas import InputType
from services.base_service import BaseService, ServiceConfig
from input_handlers import FrameInputPacket
from core.utils import ensure_model_available
from models.yolo_model import YOLOModel, load_yolo_model

logger = logging.getLogger(__name__)


# YOLO Pose keypoint names (COCO format)
YOLO_POSE_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


@dataclass
class PoseServiceConfig(ServiceConfig):
    """
    Configuration for Pose Estimation Service.

    Extends base ServiceConfig with model-specific settings.
    """
    # Model settings
    model_id: str = ""
    model_url: str = ""  # URL to download model from
    model_path: str = ""  # Local path (will be set after download)
    model_architecture: str = "yolov8-pose"

    # Inference settings
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    half_precision: bool = False

    # Keypoint settings
    keypoint_confidence_threshold: float = 0.5

    # Class mapping (usually just person class for pose)
    class_mapping: Dict[int, str] = field(default_factory=lambda: {0: "PERSON"})

    @classmethod
    def from_game_config(
        cls,
        match_id: str,
        input_source: str,
        model_config: Dict[str, Any],
        service_config: Dict[str, Any],
        inference_settings: Dict[str, Any],
        input_type: InputType = InputType.HLS
    ) -> "PoseServiceConfig":
        """
        Create PoseServiceConfig from GameConfig components.
        """
        model_params = model_config.get("params", {})

        return cls(
            match_id=match_id,
            service_id=f"pose_{model_config.get('model_id', 'unknown')}",
            input_source=input_source,
            input_type=input_type,
            target_width=inference_settings.get("processing_resolution", [1280, 720])[0],
            target_height=inference_settings.get("processing_resolution", [1280, 720])[1],
            frame_skip=inference_settings.get("frame_skip", 1),
            device=service_config.get("device", "cuda:0"),
            model_id=model_config.get("model_id", ""),
            model_url=model_config.get("model_url", ""),
            model_architecture=model_config.get("model_architecture", "yolov8-pose"),
            confidence_threshold=model_params.get("confidence_threshold", 0.5),
            iou_threshold=model_params.get("iou_threshold", 0.45),
            max_detections=model_params.get("max_detections", 100),
            half_precision=model_params.get("half_precision", False),
            keypoint_confidence_threshold=service_config.get("keypoint_confidence_threshold", 0.5),
        )


class PoseService(BaseService):
    """
    Pose Estimation Service.

    Runs YOLO-Pose (or other) pose estimation models on video frames.
    Outputs keypoints for each detected person.

    Usage:
        config = PoseServiceConfig(
            match_id="match_123",
            service_id="pose_yolov8",
            stream_url="https://example.com/stream.m3u8",
            model_id="yolov8n-pose",
            model_path="/path/to/yolov8n-pose.pt",
            device="cuda:0",
        )
        service = PoseService(config)
        service.run()  # Blocking - processes until stream ends
    """

    def __init__(self, config: PoseServiceConfig):
        super().__init__(config)
        self.pose_config = config
        self._model: Optional[YOLOModel] = None

    def initialize(self) -> bool:
        """
        Initialize Pose service - download and load model.

        Returns:
            True if initialization successful
        """
        try:
            # Determine local model path
            if self.pose_config.model_path:
                local_path = self.pose_config.model_path
            else:
                # Default: /tmp/models/{model_id}/latest.pt
                model_dir = os.path.join("/tmp", "models", self.pose_config.model_id)
                os.makedirs(model_dir, exist_ok=True)
                local_path = os.path.join(model_dir, "latest.pt")

            # Ensure model is available (download if needed)
            if self.pose_config.model_url and not os.path.exists(local_path):
                logger.info(f"Downloading model from {self.pose_config.model_url}")
                local_path = ensure_model_available(
                    self.pose_config.model_url,
                    local_path
                )

            # Load model
            logger.info(f"Loading pose model from {local_path}")
            self._model = load_yolo_model(
                model_path=local_path,
                device=self.pose_config.device,
                half_precision=self.pose_config.half_precision,
                warmup=True
            )

            logger.info(f"Pose Service initialized: model={self.pose_config.model_id}, device={self.pose_config.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Pose service: {e}")
            return False

    def process_frame(self, frame_packet: FrameInputPacket) -> Optional[Dict[str, Any]]:
        """
        Process a single frame - run pose estimation.

        Args:
            frame_packet: Frame data and metadata from input handler

        Returns:
            Dict with pose detections for DB storage
        """
        try:
            # Run inference
            outputs = self._model.predict(
                [frame_packet.frame],
                confidence=self.pose_config.confidence_threshold,
                iou=self.pose_config.iou_threshold,
                max_det=self.pose_config.max_detections,
            )

            if not outputs:
                return None

            output = outputs[0]

            # Extract pose data from YOLO results
            poses = self._extract_poses(output)

            return {
                "model_id": self.pose_config.model_id,
                "poses": poses,
            }

        except Exception as e:
            logger.error(f"Error processing frame {frame_packet.sequence_number}: {e}")
            return None

    def _extract_poses(self, output) -> List[Dict[str, Any]]:
        """
        Extract pose keypoints from YOLO output.

        Args:
            output: YOLO model output

        Returns:
            List of pose dictionaries
        """
        poses = []

        # YOLO pose output contains keypoints in the results
        if hasattr(output, 'keypoints') and output.keypoints is not None:
            keypoints_data = output.keypoints
            boxes_data = output.boxes if hasattr(output, 'boxes') else None

            # Get frame dimensions for normalization
            orig_shape = output.orig_shape if hasattr(output, 'orig_shape') else (1, 1)
            h, w = orig_shape[0], orig_shape[1]

            for i, kpts in enumerate(keypoints_data.data):
                pose = {
                    "person_id": i,
                    "confidence": float(boxes_data.conf[i]) if boxes_data is not None and hasattr(boxes_data, 'conf') else 1.0,
                    "keypoints": []
                }

                # Add bounding box if available
                if boxes_data is not None and hasattr(boxes_data, 'xyxyn'):
                    box = boxes_data.xyxyn[i]
                    pose["bbox"] = {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }

                # Extract keypoints
                for j, kpt in enumerate(kpts):
                    x, y = float(kpt[0]), float(kpt[1])
                    conf = float(kpt[2]) if len(kpt) > 2 else 1.0

                    # Normalize coordinates
                    x_norm = x / w if w > 0 else x
                    y_norm = y / h if h > 0 else y

                    # Only include keypoints above threshold
                    if conf >= self.pose_config.keypoint_confidence_threshold:
                        keypoint = {
                            "x": x_norm,
                            "y": y_norm,
                            "confidence": conf,
                            "name": YOLO_POSE_KEYPOINTS[j] if j < len(YOLO_POSE_KEYPOINTS) else f"keypoint_{j}"
                        }
                        pose["keypoints"].append(keypoint)

                poses.append(pose)

        return poses

    def cleanup(self):
        """Clean up model resources"""
        if self._model:
            self._model = None
        logger.info("Pose Service cleaned up")


# =============================================================================
# CONFIG BUILDER (for ServiceRegistry)
# =============================================================================

def build_pose_config(
    match_id: str,
    source_url: str,
    model_config: dict,
    service_config: dict,
    inference_settings: dict,
    start_frame: int = 0,
    start_segment: int = 0,
    device: str = "cuda:0",
    local_mode: bool = False,
    output_dir: str = None,
) -> PoseServiceConfig:
    """
    Build PoseServiceConfig from game config components.

    This is the config builder registered with ServiceRegistry.
    Encapsulates all knowledge of how to construct Pose configs.

    Args:
        match_id: Match identifier
        source_url: Stream URL or file path
        model_config: Model configuration dict
        service_config: Service template dict
        inference_settings: Inference settings dict
        start_frame: Frame to start from (for resume)
        start_segment: Segment to start from (for HLS resume)
        device: Device to run on
        local_mode: If True, output to local files
        output_dir: Output directory for local mode

    Returns:
        Configured PoseServiceConfig
    """
    from core.source_router import SourceRouter

    # Detect input type using SourceRouter
    source_type = SourceRouter.detect(source_url)
    input_type = SourceRouter.to_input_type(source_type)

    # Extract model parameters
    model_params = model_config.get("default_params", {})
    processing_resolution = inference_settings.get("processing_resolution", [1280, 720])

    return PoseServiceConfig(
        # Identifiers
        match_id=match_id,
        service_id=f"pose_{model_config.get('model_id', 'unknown')}",

        # Input
        input_source=source_url,
        input_type=input_type,

        # Processing
        target_width=processing_resolution[0] if processing_resolution else None,
        target_height=processing_resolution[1] if len(processing_resolution) > 1 else None,
        frame_skip=inference_settings.get("frame_skip", 1),
        start_frame=start_frame,
        start_segment=start_segment,

        # Batching (for GPU efficiency)
        inference_batch_size=model_params.get("batch_size", 1),

        # Device
        device=device,

        # Database
        db_table_name=service_config.get("db_table_name", "inference_results"),
        db_batch_size=service_config.get("db_batch_size", 12),
        db_flush_interval_ms=service_config.get("db_flush_interval_ms", 250),

        # Local mode
        local_mode=local_mode,
        local_output_dir=output_dir,

        # Model
        model_id=model_config.get("model_id", ""),
        model_url=model_config.get("model_url", ""),
        model_path=model_config.get("model_path", ""),
        model_architecture=model_config.get("model_architecture", "yolov8-pose"),

        # Inference parameters
        confidence_threshold=model_params.get("confidence_threshold", 0.5),
        iou_threshold=model_params.get("iou_threshold", 0.45),
        max_detections=model_params.get("max_detections", 100),
        half_precision=model_params.get("half_precision", False),

        # Keypoint settings
        keypoint_confidence_threshold=service_config.get("keypoint_confidence_threshold", 0.5),
    )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for running Pose service standalone"""
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Pose Estimation Service")
    parser.add_argument("--match-id", required=True, help="Match identifier")
    parser.add_argument("--stream-url", required=True, help="Stream URL or file path")
    parser.add_argument("--input-type", default="hls", choices=["hls", "mp4", "file"], help="Input type")
    parser.add_argument("--model-id", required=True, help="Model identifier")
    parser.add_argument("--model-url", help="URL to download model from")
    parser.add_argument("--model-path", help="Local path to model file")
    parser.add_argument("--device", default="cuda:0", help="Device (cpu, cuda:0, etc.)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--keypoint-confidence", type=float, default=0.5, help="Keypoint confidence threshold")
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

    # Create config
    config = PoseServiceConfig(
        match_id=args.match_id,
        service_id=f"pose_{args.model_id}",
        input_source=args.stream_url,
        input_type=input_type,
        model_id=args.model_id,
        model_url=args.model_url or "",
        model_path=args.model_path or "",
        device=args.device,
        confidence_threshold=args.confidence,
        keypoint_confidence_threshold=args.keypoint_confidence,
        target_width=args.width,
        target_height=args.height,
        db_table_name=args.db_table,
        local_output_dir=args.local_output,
        start_frame=args.start_frame,
        start_segment=args.start_segment,
    )

    # Run service
    service = PoseService(config)
    service.run()


if __name__ == "__main__":
    main()
