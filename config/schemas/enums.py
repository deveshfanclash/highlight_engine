"""
Enums for Inference Configuration

All enumeration types used across the configuration system.
"""

from enum import Enum


class ModelType(str, Enum):
    """Types of ML models"""
    OBJECT_DETECTION = "object_detection"
    POSE_ESTIMATION = "pose_estimation"


class ModelArchitecture(str, Enum):
    """Supported model architectures"""
    YOLO_V8 = "yolov8"
    YOLO_V11 = "yolov11"
    YOLO_V12 = "yolov12"
    YOLO_POSE = "yolov8-pose"
    CUSTOM = "custom"


class ServiceType(str, Enum):
    """Types of inference services"""
    OBJECT_DETECTION = "object_detection"
    POSE_ESTIMATION = "pose_estimation"
    REPLAY_DETECTION = "replay_detection"


class InputType(str, Enum):
    """Types of input sources"""
    HLS = "hls"
    RTSP = "rtsp"
    MP4 = "mp4"
    FILE = "file"


class ProcessingPattern(str, Enum):
    """Processing patterns for services"""
    FRAME_BY_FRAME = "frame_by_frame"
    FRAME_BATCHES = "frame_batches"


class DeviceType(str, Enum):
    """Device types for inference"""
    CPU = "cpu"
    GPU = "gpu"
    MPS = "mps"


class GameCategory(str, Enum):
    """Categories of games/sports"""
    BALL_SPORT = "ball_sport"
    RACKET_SPORT = "racket_sport"
    COMBAT_SPORT = "combat_sport"
    ATHLETICS = "athletics"
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
    KEYPOINTS = "keypoints"
