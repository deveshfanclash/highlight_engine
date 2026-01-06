"""
Input Handlers

Decision layer for how services receive input data.
Services choose handlers based on their processing pattern.

Handlers:
- FrameInputHandler: Frame-by-frame video processing
- (Future) ClipInputHandler: Clip-based video processing
- (Future) AudioInputHandler: Audio stream processing
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

__all__ = [
    # Base
    "BaseInputHandler",
    "InputPacket",
    "get_handler_for_pattern",
    # Frame
    "FrameInputHandler",
    "FrameInputPacket",
]
