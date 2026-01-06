"""
Enums for Inference Configuration

All enumeration types used across the configuration system.
"""

from enum import Enum


class ModelType(str, Enum):
    """Types of ML models"""
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    POSE_ESTIMATION = "pose_estimation"
    CLASSIFICATION = "classification"
    AUDIO = "audio"
    VLM = "vlm"  # Vision-Language Model


class ModelArchitecture(str, Enum):
    """Supported model architectures"""
    YOLO_V8 = "yolov8"
    YOLO_V11 = "yolov11"
    YOLO_V12 = "yolov12"
    YOLO_SEG = "yolov8-seg"
    YOLO_POSE = "yolov8-pose"
    RF_DETR = "rf_detr"
    SAM = "sam"
    WHISPER = "whisper"
    GEMINI = "gemini"
    CUSTOM = "custom"


class ServiceType(str, Enum):
    """Types of inference services"""
    OBJECT_DETECTION = "object_detection"
    CAMERA_VIEW = "camera_view"
    SEGMENTATION = "segmentation"
    POSE_ESTIMATION = "pose_estimation"
    REPLAY_DETECTION = "replay_detection"
    EVENT_DETECTION = "event_detection"
    AUDIO_ANALYSIS = "audio_analysis"
    HLS_METADATA = "hls_metadata"


class InputType(str, Enum):
    """Types of input sources"""
    HLS = "hls"
    RTSP = "rtsp"
    MP4 = "mp4"
    FILE = "file"
    AUDIO = "audio"
    CLIP = "clip"


class ProcessingPattern(str, Enum):
    """Processing patterns for services"""
    FRAME_BY_FRAME = "frame_by_frame"
    CLIP_BASED = "clip_based"
    AUDIO_BASED = "audio_based"
    FULL_VIDEO = "full_video"


class DeviceType(str, Enum):
    """Device types for inference"""
    CPU = "cpu"
    GPU = "gpu"
    # Specific GPU can be "cuda:0", "cuda:1", etc.


class GameCategory(str, Enum):
    """Categories of games/sports"""
    BALL_SPORT = "ball_sport"
    RACKET_SPORT = "racket_sport"
    COMBAT_SPORT = "combat_sport"
    ATHLETICS = "athletics"
    ESPORTS = "esports"
    OTHER = "other"


class MatchStatus(str, Enum):
    """Status of match processing"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class OutputFormat(str, Enum):
    """Output formats for model results"""
    BBOX = "bbox"
    BBOX_WITH_MASK = "bbox_with_mask"
    MASK = "mask"
    KEYPOINTS = "keypoints"
    EMBEDDING = "embedding"
    TEXT = "text"
