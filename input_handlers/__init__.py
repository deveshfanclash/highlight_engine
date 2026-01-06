"""
Input Handlers

Decision layer for how services receive input data.
Services choose handlers based on their processing pattern.

Handlers:
- FrameInputHandler: Frame-by-frame video processing (MP4, HLS, RTSP)
- AudioInputHandler: Audio stream processing (for Whisper, etc.)
- (Future) ClipInputHandler: Clip-based video processing
"""

from input_handlers.base import (
    BaseInputHandler,
    InputPacket,
    get_handler_for_pattern,
)
from input_handlers.frame_handler import (
    FrameInputHandler,
    FrameInputPacket,
)
from input_handlers.audio_handler import (
    AudioInputHandler,
    AudioInputPacket,
)

__all__ = [
    # Base
    "BaseInputHandler",
    "InputPacket",
    "get_handler_for_pattern",
    # Frame
    "FrameInputHandler",
    "FrameInputPacket",
    # Audio
    "AudioInputHandler",
    "AudioInputPacket",
]
