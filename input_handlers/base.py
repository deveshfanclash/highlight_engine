"""
Base Input Handler

Abstract base class for all input handlers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Any, Dict
import logging

from config.schemas import InputType

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
    based on its processing pattern.
    """

    def __init__(
        self,
        input_source: str,
        input_type: InputType,
        **kwargs
    ):
        self.input_source = input_source
        self.input_type = input_type
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
