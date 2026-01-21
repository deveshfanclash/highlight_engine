"""
Core Module

Contains foundational components used across all services:
- Frame extraction from streams
- Common utilities
- Data structures
- Resume logic
"""

from core.frame_provider import FrameProvider, FramePacket
from core.utils import (
    frame_to_timecode,
    get_video_resolution_and_fps,
    get_best_stream_url,
    download_file,
    normalize_bbox,
    denormalize_bbox,
)
from core.resume import (
    ResumeMode,
    ResumePosition,
    get_resume_position,
)

__all__ = [
    "FrameProvider",
    "FramePacket",
    "frame_to_timecode",
    "get_video_resolution_and_fps",
    "get_best_stream_url",
    "download_file",
    "normalize_bbox",
    "denormalize_bbox",
    "ResumeMode",
    "ResumePosition",
    "get_resume_position",
]
