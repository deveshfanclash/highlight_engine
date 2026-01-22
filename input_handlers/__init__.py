"""
Input Handlers

Decision layer for how services receive input data.
Services choose handlers based on their processing pattern.

Handlers:
- FrameInputHandler: Frame-by-frame video processing (MP4, HLS, RTSP)

Buffer Modes:
- BufferMode.FIFO: Process all frames in order (VOD)
- BufferMode.DROP_OLD: Drop old frames when full (real-time)
"""

from input_handlers.base import (
    BaseInputHandler,
    InputPacket,
)
from input_handlers.frame_handler import (
    FrameInputHandler,
    FrameInputPacket,
)
from core.stream_buffer import BufferMode

__all__ = [
    "BaseInputHandler",
    "InputPacket",
    "FrameInputHandler",
    "FrameInputPacket",
    "BufferMode",
]
