"""
Core Module

Contains foundational components used across all services:
- Frame extraction from streams
- Source type routing
- Batch accumulation for GPU efficiency
- Stream buffering for async processing
- Common utilities
- Resume logic
"""

from core.frame_provider import FrameProvider, FramePacket
from core.source_router import SourceRouter, SourceType, detect_source_type, needs_ffmpeg
from core.batch_accumulator import BatchAccumulator, Batch
from core.stream_buffer import StreamBuffer, BufferedFrameProvider, BufferMode
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
    # Frame extraction
    "FrameProvider",
    "FramePacket",
    # Source routing
    "SourceRouter",
    "SourceType",
    "detect_source_type",
    "needs_ffmpeg",
    # Batching
    "BatchAccumulator",
    "Batch",
    # Stream buffering
    "StreamBuffer",
    "BufferedFrameProvider",
    "BufferMode",
    # Utilities
    "frame_to_timecode",
    "get_video_resolution_and_fps",
    "get_best_stream_url",
    "download_file",
    "normalize_bbox",
    "denormalize_bbox",
    # Resume
    "ResumeMode",
    "ResumePosition",
    "get_resume_position",
]
