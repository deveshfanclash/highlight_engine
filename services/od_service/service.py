"""
Object Detection Service

Runs YOLO object detection on video frames.
Returns typed DetectionResults for type safety.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from config.schemas import InputType
from services.base_service import BaseService, ServiceConfig
from input_handlers import FrameInputPacket
from models.yolo_model import YOLOModel, load_yolo_model
from models.downloader import download_model
from output.results import DetectionResults

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

    Returns typed DetectionResults with filtering and serialization support.

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
                # Download using ModelDownloader
                model_id = self.od_config.model_id or "default"
                local_path = str(download_model(
                    source=self.od_config.model_url,
                    model_id=model_id,
                    filename="model.pt"
                ))
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

    @property
    def supports_batching(self) -> bool:
        """ODService supports efficient batched processing."""
        return True

    @property
    def supports_multi_worker(self) -> bool:
        """ODService supports multi-worker parallel processing."""
        return True

    def get_model_path(self) -> str:
        """Get path to model file for worker initialization."""
        if self.od_config.model_path:
            return self.od_config.model_path
        elif self.od_config.model_url:
            model_id = self.od_config.model_id or "default"
            return str(download_model(
                source=self.od_config.model_url,
                model_id=model_id,
                filename="model.pt"
            ))
        raise ValueError("No model_url or model_path specified")

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for worker initialization."""
        return {
            "model_id": self.od_config.model_id,
            "confidence_threshold": self.od_config.confidence_threshold,
            "iou_threshold": self.od_config.iou_threshold,
            "max_detections": self.od_config.max_detections,
            "half_precision": self.od_config.half_precision,
            "classes_to_detect": self.od_config.classes_to_detect,
        }

    @staticmethod
    def worker_init(
        device: str,
        model_path: str,
        half_precision: bool = False,
        classes_to_detect: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initialize YOLO model in worker process.

        Args:
            device: Device to load model on
            model_path: Path to model file
            half_precision: Use FP16 inference
            classes_to_detect: Class names to filter

        Returns:
            Dict with model and class IDs
        """
        model = load_yolo_model(
            model_path=model_path,
            device=device,
            half_precision=half_precision,
            warmup=True,
        )

        # Convert class names to IDs
        class_ids = []
        if classes_to_detect:
            class_ids = model.get_class_ids_for_names(classes_to_detect)

        return {
            "model": model,
            "class_ids": class_ids,
            "config": kwargs,
        }

    @staticmethod
    def worker_process_batch(
        model: Dict[str, Any],
        frames: List,
        metadata: List[Dict[str, Any]],
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process batch of frames in worker process.

        Args:
            model: Dict from worker_init containing model and config
            frames: List of numpy arrays
            metadata: List of metadata dicts

        Returns:
            List of result dicts
        """
        yolo_model = model["model"]
        class_ids = model["class_ids"]
        config = model.get("config", {})

        confidence = config.get("confidence_threshold", 0.5)
        iou = config.get("iou_threshold", 0.45)
        max_det = config.get("max_detections", 100)
        model_id = config.get("model_id", "")

        try:
            results = yolo_model.predict(
                frames,
                confidence=confidence,
                classes=class_ids if class_ids else None,
                iou=iou,
                max_det=max_det,
            )

            output = []
            for detection_result in results:
                if detection_result is None:
                    output.append(None)
                else:
                    output.append({
                        "model_id": model_id,
                        "detections": [d.to_dict() for d in detection_result.detections],
                        "detection_count": detection_result.count,
                    })

            return output

        except Exception as e:
            logger.error(f"Worker error processing batch: {e}")
            return [None] * len(frames)

    def process_frame(self, frame_packet: FrameInputPacket) -> Optional[Dict[str, Any]]:
        """Process a single frame and return dict for DB storage."""
        try:
            results = self._model.predict(
                [frame_packet.frame],
                confidence=self.od_config.confidence_threshold,
                classes=self._class_ids if self._class_ids else None,
                iou=self.od_config.iou_threshold,
                max_det=self.od_config.max_detections,
            )

            if not results:
                return None

            # Convert typed DetectionResults to dict for storage
            detection_result = results[0]
            return {
                "model_id": self.od_config.model_id,
                "detections": [d.to_dict() for d in detection_result.detections],
                "detection_count": detection_result.count,
            }

        except Exception as e:
            logger.error(f"Error processing frame {frame_packet.sequence_number}: {e}")
            return None

    def process_batch(
        self,
        frame_packets: List[FrameInputPacket]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process a batch of frames efficiently.

        This runs a single model.predict() call with multiple frames,
        which is significantly faster than processing frames one at a time
        (GPU parallelism).
        """
        try:
            # Extract frames from packets
            frames = [pkt.frame for pkt in frame_packets]

            # Run batch inference - returns List[DetectionResults]
            results = self._model.predict(
                frames,
                confidence=self.od_config.confidence_threshold,
                classes=self._class_ids if self._class_ids else None,
                iou=self.od_config.iou_threshold,
                max_det=self.od_config.max_detections,
            )

            # Convert DetectionResults to dicts for storage
            output = []
            for detection_result in results:
                if detection_result is None:
                    output.append(None)
                else:
                    output.append({
                        "model_id": self.od_config.model_id,
                        "detections": [d.to_dict() for d in detection_result.detections],
                        "detection_count": detection_result.count,
                    })

            return output

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return None for all frames in batch on error
            return [None] * len(frame_packets)

    def cleanup(self):
        """Clean up resources."""
        self._model = None
        logger.info("OD Service cleaned up")


# =============================================================================
# CONFIG BUILDER (for ServiceRegistry)
# =============================================================================

def build_od_config(
    match_id: str,
    source_url: str,
    model_config: dict,
    service_config: dict,
    inference_settings: dict,
    start_frame: int = 0,
    start_segment: int = 0,
    local_mode: bool = False,
    output_dir: str = None,
) -> ODServiceConfig:
    """
    Build ODServiceConfig from game config components.

    This is the config builder registered with ServiceRegistry.
    Encapsulates all knowledge of how to construct OD configs.

    Args:
        match_id: Match identifier
        source_url: Stream URL or file path
        model_config: Model configuration dict
        service_config: Service template dict (includes device)
        inference_settings: Inference settings dict
        start_frame: Frame to start from (for resume)
        start_segment: Segment to start from (for HLS resume)
        local_mode: If True, output to local files
        output_dir: Output directory for local mode

    Returns:
        Configured ODServiceConfig
    """
    from core.source_router import SourceRouter

    # Detect input type using SourceRouter
    source_type = SourceRouter.detect(source_url)
    input_type = SourceRouter.to_input_type(source_type)

    # Extract model parameters
    model_params = model_config.get("default_params", {})
    processing_resolution = inference_settings.get("processing_resolution", [1280, 720])

    return ODServiceConfig(
        # Identifiers
        match_id=match_id,
        service_id=f"od_{model_config.get('model_id', 'unknown')}",

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

        # Multi-worker settings
        num_workers=inference_settings.get("num_workers", 1),
        worker_queue_size=inference_settings.get("worker_queue_size", 0),

        # Buffer settings
        enable_buffering=inference_settings.get("enable_buffering", None),
        buffer_size=inference_settings.get("buffer_size", 30),
        buffer_mode=inference_settings.get("buffer_mode", "drop_old"),

        # Device (from service config)
        device=service_config.get("device", "cuda:0"),

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

        # Inference parameters
        confidence_threshold=model_params.get("confidence_threshold", 0.5),
        iou_threshold=model_params.get("iou_threshold", 0.45),
        max_detections=model_params.get("max_detections", 100),
        half_precision=model_params.get("half_precision", False),

        # Classes
        classes_to_detect=model_config.get("classes_to_detect", []),
    )
