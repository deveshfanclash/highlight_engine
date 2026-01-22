"""
Frame Input Handler

Provides frame-by-frame input handling for video sources.
Wraps FrameProvider for a service-oriented interface.
Supports optional buffering for async processing via StreamBuffer.
"""

from dataclasses import dataclass
from typing import Iterator, Optional, Union
import logging

import numpy as np

from config.schemas import InputType, ProcessingPattern
from core.frame_provider import FrameProvider, FrameProviderConfig, StreamType, FramePacket
from core.stream_buffer import StreamBuffer, BufferedFrameProvider, BufferMode
from core.source_router import SourceRouter
from input_handlers.base import BaseInputHandler, InputPacket

logger = logging.getLogger(__name__)


@dataclass
class FrameInputPacket(InputPacket):
    """
    Input packet containing a video frame.

    Extends InputPacket with frame-specific fields.
    """
    # Frame data
    frame: np.ndarray = None
    width: int = 0
    height: int = 0

    # Convenience: data points to frame
    def __post_init__(self):
        super().__post_init__()
        if self.frame is not None:
            self.data = self.frame


def _input_type_to_stream_type(input_type: InputType) -> StreamType:
    """Map InputType enum to StreamType enum"""
    mapping = {
        InputType.HLS: StreamType.HLS,
        InputType.RTSP: StreamType.RTSP,
        InputType.MP4: StreamType.MP4,
        InputType.FILE: StreamType.FILE,
    }
    return mapping.get(input_type, StreamType.FILE)


class FrameInputHandler(BaseInputHandler):
    """
    Frame-by-frame input handler for video sources.

    Wraps FrameProvider to provide a consistent interface for services
    that need frame-by-frame processing (object detection, camera view, etc.)

    Supports optional buffering for decoupling frame extraction from processing:
    - buffered=False: Direct iteration, blocking
    - buffered=True: Background frame extraction with StreamBuffer
    - buffered=None (default): Auto-detect based on source type
      - HLS/RTSP streams: Enable buffering with DROP_OLD mode (real-time)
      - Local files: Disable buffering (VOD)

    Usage:
        # Auto-detect mode (default) - streams get buffered, files don't
        handler = FrameInputHandler(
            input_source="https://example.com/stream.m3u8",
            input_type=InputType.HLS,
        )

        # Explicit buffered mode for real-time streams
        handler = FrameInputHandler(
            input_source="rtsp://camera/stream",
            input_type=InputType.RTSP,
            buffered=True,
            buffer_size=30,
            buffer_mode=BufferMode.DROP_OLD,  # Real-time: drop old frames
        )

        # Explicit direct mode
        handler = FrameInputHandler(
            input_source="/path/to/video.mp4",
            input_type=InputType.MP4,
            buffered=False,
        )

        for packet in handler.iterate():
            process_frame(packet.frame)
    """

    def __init__(
        self,
        input_source: str,
        input_type: InputType,
        processing_pattern: ProcessingPattern = ProcessingPattern.FRAME_BY_FRAME,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        frame_skip: int = 1,
        start_frame: int = 0,
        start_segment: int = 1,
        resolution_preference: str = "_1080p.m3u8",
        # Buffer settings
        buffered: Optional[bool] = None,  # None = auto-detect based on source type
        buffer_size: int = 30,
        buffer_mode: Union[BufferMode, str] = BufferMode.FIFO,
        **kwargs
    ):
        super().__init__(
            input_source=input_source,
            input_type=input_type,
            processing_pattern=processing_pattern,
            **kwargs
        )

        self.target_width = target_width
        self.target_height = target_height
        self.frame_skip = frame_skip
        self.start_frame = start_frame
        self.start_segment = start_segment
        self.resolution_preference = resolution_preference

        # Handle buffer_mode as string or enum
        if isinstance(buffer_mode, str):
            buffer_mode = BufferMode.DROP_OLD if buffer_mode == "drop_old" else BufferMode.FIFO

        # Auto-detect buffering based on source type
        if buffered is None:
            source_type = SourceRouter.detect(input_source)
            if SourceRouter.is_stream(source_type):
                # HLS or RTSP: enable buffering with DROP_OLD for real-time
                self.buffered = True
                self.buffer_mode = BufferMode.DROP_OLD
                logger.info(f"Auto-enabled buffering for stream source: {source_type.value}")
            else:
                # Local files: no buffering needed
                self.buffered = False
                self.buffer_mode = buffer_mode
        else:
            self.buffered = buffered
            self.buffer_mode = buffer_mode

        self.buffer_size = buffer_size

        # Frame provider instance
        self._provider: Optional[FrameProvider] = None
        self._buffered_provider: Optional[BufferedFrameProvider] = None

    def initialize(self) -> bool:
        """Initialize the frame provider and detect stream metadata."""
        try:
            # Map input type to stream type
            stream_type = _input_type_to_stream_type(self.input_type)

            # Create frame provider config
            config = FrameProviderConfig(
                stream_url=self.input_source,
                stream_type=stream_type,
                target_width=self.target_width,
                target_height=self.target_height,
                resolution_preference=self.resolution_preference,
                start_frame=self.start_frame,
                start_segment=self.start_segment,
                frame_skip=self.frame_skip,
            )

            # Create and initialize provider
            self._provider = FrameProvider(config)
            success = self._provider.initialize()

            if success:
                # Store metadata
                self.metadata = {
                    "source_width": self._provider.source_width,
                    "source_height": self._provider.source_height,
                    "effective_width": self._provider.effective_width,
                    "effective_height": self._provider.effective_height,
                    "fps": self._provider.fps,
                    "buffered": self.buffered,
                    "buffer_size": self.buffer_size if self.buffered else None,
                }
                self._initialized = True
                logger.info(f"FrameInputHandler initialized: {self.metadata}")

            return success

        except Exception as e:
            logger.error(f"Failed to initialize FrameInputHandler: {e}")
            return False

    def iterate(self) -> Iterator[FrameInputPacket]:
        """
        Generate frame packets for processing.

        Uses direct iteration or buffered iteration based on config.

        Yields:
            FrameInputPacket objects containing frame data
        """
        if not self._initialized:
            if not self.initialize():
                return

        self._running = True

        if self.buffered:
            yield from self._iterate_buffered()
        else:
            yield from self._iterate_direct()

    def _iterate_direct(self) -> Iterator[FrameInputPacket]:
        """Direct iteration without buffering."""
        try:
            for frame_packet in self._provider.frames():
                if not self._running:
                    break

                yield self._convert_packet(frame_packet)

        except GeneratorExit:
            logger.info("Frame iteration stopped by consumer")
        finally:
            self._running = False

    def _iterate_buffered(self) -> Iterator[FrameInputPacket]:
        """Buffered iteration with background frame extraction."""
        try:
            # Create buffered provider
            self._buffered_provider = BufferedFrameProvider(
                frame_source=self._provider.frames(),
                buffer_size=self.buffer_size,
                mode=self.buffer_mode,
            )
            self._buffered_provider.start()

            logger.info(
                f"Buffered frame iteration started "
                f"(size={self.buffer_size}, mode={self.buffer_mode.value})"
            )

            # Iterate from buffer
            for frame_packet in self._buffered_provider.iterate():
                if not self._running:
                    break

                yield self._convert_packet(frame_packet)

        except GeneratorExit:
            logger.info("Buffered frame iteration stopped by consumer")
        finally:
            if self._buffered_provider:
                stats = self._buffered_provider.stats
                logger.info(f"Buffer stats: {stats}")
                self._buffered_provider.stop()
                self._buffered_provider = None
            self._running = False

    def _convert_packet(self, frame_packet: FramePacket) -> FrameInputPacket:
        """Convert FramePacket to FrameInputPacket."""
        return FrameInputPacket(
            sequence_number=frame_packet.frame_number,
            timestamp_ms=frame_packet.timestamp_ms,
            data=frame_packet.frame,
            frame=frame_packet.frame,
            width=frame_packet.width,
            height=frame_packet.height,
            segment_number=frame_packet.segment_number,
        )

    def stop(self):
        """Stop frame extraction and cleanup."""
        self._running = False

        if self._buffered_provider:
            self._buffered_provider.stop()
            self._buffered_provider = None

        if self._provider:
            self._provider.stop()
            self._provider = None

        logger.info("FrameInputHandler stopped")

    # Buffer statistics
    @property
    def buffer_stats(self) -> Optional[dict]:
        """Get buffer statistics (if buffered mode)."""
        if self._buffered_provider:
            return self._buffered_provider.stats
        return None

    # Convenience properties
    @property
    def fps(self) -> Optional[float]:
        """Get detected FPS"""
        return self.metadata.get("fps")

    @property
    def effective_width(self) -> Optional[int]:
        """Get effective processing width"""
        return self.metadata.get("effective_width")

    @property
    def effective_height(self) -> Optional[int]:
        """Get effective processing height"""
        return self.metadata.get("effective_height")
