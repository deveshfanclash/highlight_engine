"""
Core Utilities Module

Re-exports utilities for convenience:
- utils: Common helper functions (timecode, video info, bbox operations)
- output_utils: Typed result classes (Detection, DetectionResults, etc.)
"""

from core.utils.utils import (
    frame_to_timecode,
    frame_to_ms,
    get_video_resolution_and_fps,
    get_best_stream_url,
    download_file,
    normalize_bbox,
    denormalize_bbox,
)

from core.utils.output_utils import (
    BoxFormat,
    BoundingBox,
    Detection,
    DetectionResults,
    Keypoint,
    KeypointSkeleton,
    PoseResults,
    create_detection_results,
)

__all__ = [
    # Utils
    "frame_to_timecode",
    "frame_to_ms",
    "get_video_resolution_and_fps",
    "get_best_stream_url",
    "download_file",
    "normalize_bbox",
    "denormalize_bbox",
    # Output utils / Results
    "BoxFormat",
    "BoundingBox",
    "Detection",
    "DetectionResults",
    "Keypoint",
    "KeypointSkeleton",
    "PoseResults",
    "create_detection_results",
]
