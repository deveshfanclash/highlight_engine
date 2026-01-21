"""
Input Handlers

Decision layer for how services receive input data.
Services choose handlers based on their processing pattern.

Handlers:
- FrameInputHandler: Frame-by-frame video processing (MP4, HLS, RTSP)
"""

from input_handlers.base import (
    BaseInputHandler,
    InputPacket,
)
from input_handlers.frame_handler import (
    FrameInputHandler,
    FrameInputPacket,
)

__all__ = [
    "BaseInputHandler",
    "InputPacket",
    "FrameInputHandler",
    "FrameInputPacket",
]
