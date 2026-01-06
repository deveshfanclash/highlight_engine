"""
Audio Input Handler

Provides audio-based input handling for audio analysis services.
"""

import subprocess
import logging
from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np

from config.schemas import InputType, ProcessingPattern
from input_handlers.base import BaseInputHandler, InputPacket

logger = logging.getLogger(__name__)


@dataclass
class AudioInputPacket(InputPacket):
    """
    Input packet containing audio samples.
    """
    audio: np.ndarray = None  # Audio samples
    sample_rate: int = 16000
    channels: int = 1
    duration_ms: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.audio is not None:
            self.data = self.audio


class AudioInputHandler(BaseInputHandler):
    """
    Audio input handler for audio analysis services.

    Extracts audio from video files or audio streams and yields
    audio chunks for processing.

    Usage:
        handler = AudioInputHandler(
            input_source="/path/to/video.mp4",
            input_type=InputType.MP4,
            processing_pattern=ProcessingPattern.AUDIO_BASED,
            sample_rate=16000,
            chunk_duration_ms=1000,
        )

        for packet in handler.iterate():
            transcription = whisper.transcribe(packet.audio)
    """

    def __init__(
        self,
        input_source: str,
        input_type: InputType,
        processing_pattern: ProcessingPattern = ProcessingPattern.AUDIO_BASED,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 1000,  # 1 second chunks
        **kwargs
    ):
        super().__init__(
            input_source=input_source,
            input_type=input_type,
            processing_pattern=processing_pattern,
            **kwargs
        )
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms

        self._process: Optional[subprocess.Popen] = None
        self._chunk_number = 0

    def initialize(self) -> bool:
        """Initialize FFmpeg audio extraction."""
        try:
            # Build FFmpeg command for audio extraction
            cmd = [
                "ffmpeg",
                "-i", self.input_source,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-ar", str(self.sample_rate),
                "-ac", str(self.channels),
                "-f", "s16le",  # Raw PCM output
                "-"
            ]

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            self.metadata = {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "chunk_duration_ms": self.chunk_duration_ms,
            }
            self._initialized = True
            logger.info(f"AudioInputHandler initialized: {self.metadata}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AudioInputHandler: {e}")
            return False

    def iterate(self) -> Iterator[AudioInputPacket]:
        """
        Generate audio packets for processing.

        Yields:
            AudioInputPacket objects containing audio samples
        """
        if not self._initialized:
            if not self.initialize():
                return

        self._running = True

        # Calculate bytes per chunk
        bytes_per_sample = 2  # 16-bit = 2 bytes
        samples_per_chunk = int(self.sample_rate * self.chunk_duration_ms / 1000)
        bytes_per_chunk = samples_per_chunk * self.channels * bytes_per_sample

        try:
            while self._running:
                # Read chunk of audio data
                audio_bytes = self._process.stdout.read(bytes_per_chunk)

                if not audio_bytes:
                    logger.info("End of audio stream")
                    break

                # Convert to numpy array
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                # Reshape for multi-channel
                if self.channels > 1:
                    audio_array = audio_array.reshape(-1, self.channels)

                # Convert to float32 normalized
                audio_float = audio_array.astype(np.float32) / 32768.0

                timestamp_ms = self._chunk_number * self.chunk_duration_ms

                packet = AudioInputPacket(
                    sequence_number=self._chunk_number,
                    timestamp_ms=timestamp_ms,
                    data=audio_float,
                    audio=audio_float,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    duration_ms=self.chunk_duration_ms,
                )

                self._chunk_number += 1
                yield packet

        except GeneratorExit:
            logger.info("Audio iteration stopped by consumer")
        finally:
            self._running = False

    def stop(self):
        """Stop audio extraction and cleanup."""
        self._running = False
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None
        logger.info("AudioInputHandler stopped")
