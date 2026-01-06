"""
Base Input Handler

Abstract base class for all input handlers.
Services choose their input handler based on processing pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Any, Dict
import logging

from config.schemas import InputType, ProcessingPattern

logger = logging.getLogger(__name__)


@dataclass
class InputPacket:
    """
    Generic container for input data.

    Each handler type creates specific packet types, but all share:
    - sequence_number: Ordering (frame number, chunk number, etc.)
    - timestamp_ms: Time position in source
    - data: The actual input data (frame, audio, clip)
    """
    sequence_number: int
    timestamp_ms: int
    data: Any

    # Optional metadata
    segment_number: Optional[int] = None
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class BaseInputHandler(ABC):
    """
    Abstract base class for input handlers.

    Input handlers are the DECISION LAYER between raw input sources
    and services. Each service type chooses the appropriate handler
    based on its processing pattern:

    - FRAME_BY_FRAME -> FrameInputHandler
    - CLIP_BASED -> ClipInputHandler (future)
    - AUDIO_BASED -> AudioInputHandler (future)

    This abstraction allows services to remain input-agnostic.
    """

    def __init__(
        self,
        input_source: str,
        input_type: InputType,
        processing_pattern: ProcessingPattern,
        **kwargs
    ):
        self.input_source = input_source
        self.input_type = input_type
        self.processing_pattern = processing_pattern
        self.config = kwargs

        # State
        self._initialized = False
        self._running = False

        # Metadata (populated on initialize)
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the handler and detect input metadata.

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    def iterate(self) -> Iterator[InputPacket]:
        """
        Generate input packets for processing.

        Yields:
            InputPacket objects
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop processing and cleanup resources."""
        pass

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_running(self) -> bool:
        return self._running

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False


def get_handler_for_pattern(
    processing_pattern: ProcessingPattern,
    input_source: str,
    input_type: InputType,
    **kwargs
) -> BaseInputHandler:
    """
    Factory function to get the appropriate handler for a processing pattern.

    Args:
        processing_pattern: How the service processes input
        input_source: URL or path to input
        input_type: Type of input source
        **kwargs: Handler-specific configuration

    Returns:
        Appropriate input handler instance

    Raises:
        ValueError: If no handler supports the pattern
    """
    from input_handlers.frame_handler import FrameInputHandler
    from input_handlers.audio_handler import AudioInputHandler
    # Future: from input_handlers.clip_handler import ClipInputHandler

    pattern_to_handler = {
        ProcessingPattern.FRAME_BY_FRAME: FrameInputHandler,
        ProcessingPattern.AUDIO_BASED: AudioInputHandler,
        # ProcessingPattern.CLIP_BASED: ClipInputHandler,
    }

    handler_class = pattern_to_handler.get(processing_pattern)

    if handler_class is None:
        raise ValueError(f"No handler available for processing pattern: {processing_pattern}")

    return handler_class(
        input_source=input_source,
        input_type=input_type,
        processing_pattern=processing_pattern,
        **kwargs
    )
