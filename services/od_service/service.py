"""
Object Detection Service

Runs YOLO object detection on video frames.
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


@dataclass
class ODServiceConfig(ServiceConfig):
    """Configuration for Object Detection Service."""

    # Model settings
    model_id: str = ""
    model_url: str = ""
    model_path: str = ""

    # Inference settings
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    half_precision: bool = False

    # Classes to detect (user-friendly names like "ball", "person")
    # Empty list = detect all classes
    classes_to_detect: List[str] = field(default_factory=list)


class ODService(BaseService):
    """
    Object Detection Service.

    Usage:
        config = ODServiceConfig(
            match_id="match_123",
            service_id="od_football",
            input_source="/path/to/video.mp4",
            model_url="yolov8n.pt",
            classes_to_detect=["ball", "person"],
        )
        service = ODService(config)
        service.run()
    """

    def __init__(self, config: ODServiceConfig):
        super().__init__(config)
        self.od_config = config
        self._model: Optional[YOLOModel] = None
        self._class_ids: List[int] = []  # Model's native class IDs to detect

    def initialize(self) -> bool:
        """Initialize service - download and load model."""
        try:
            # Determine model path
            if self.od_config.model_path:
                local_path = self.od_config.model_path
            elif self.od_config.model_url:
                # Download if needed
                model_dir = os.path.join("/tmp", "models", self.od_config.model_id or "default")
                os.makedirs(model_dir, exist_ok=True)
                local_path = os.path.join(model_dir, "model.pt")

                if not os.path.exists(local_path):
                    logger.info(f"Downloading model: {self.od_config.model_url}")
                    local_path = ensure_model_available(self.od_config.model_url, local_path)
            else:
                logger.error("No model_url or model_path specified")
                return False

            # Load model
            logger.info(f"Loading model: {local_path}")
            self._model = load_yolo_model(
                model_path=local_path,
                device=self.od_config.device,
                half_precision=self.od_config.half_precision,
                warmup=True,
            )

            # Convert user's class names to model's native class IDs
            if self.od_config.classes_to_detect:
                self._class_ids = self._model.get_class_ids_for_names(
                    self.od_config.classes_to_detect
                )
                logger.info(
                    f"Classes to detect: {self.od_config.classes_to_detect} "
                    f"-> model IDs: {self._class_ids}"
                )
            else:
                self._class_ids = []  # Empty = all classes
                logger.info("Detecting all classes")

            logger.info(f"OD Service initialized: device={self.od_config.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    def process_frame(self, frame_packet: FrameInputPacket) -> Optional[Dict[str, Any]]:
        """Process a single frame."""
        try:
            outputs = self._model.predict(
                [frame_packet.frame],
                confidence=self.od_config.confidence_threshold,
                classes=self._class_ids if self._class_ids else None,
                iou=self.od_config.iou_threshold,
                max_det=self.od_config.max_detections,
            )

            if not outputs:
                return None

            return {
                "model_id": self.od_config.model_id,
                "detections": outputs[0].to_dict_list(),
            }

        except Exception as e:
            logger.error(f"Error processing frame {frame_packet.sequence_number}: {e}")
            return None

    def cleanup(self):
        """Clean up resources."""
        self._model = None
        logger.info("OD Service cleaned up")
