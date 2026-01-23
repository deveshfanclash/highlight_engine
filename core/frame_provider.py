"""
Frame Provider

Extracts frames from video streams (HLS, RTSP, MP4) using FFmpeg or OpenCV.
Each service gets its own FrameProvider instance for independent processing.

For local files (MP4, etc.): Uses OpenCV (faster, no subprocess)
For streams (HLS, RTSP): Uses FFmpeg

Extracted and refactored from old_inference_code_for_reference/src/live_stream_processing/real_time_inference.py
"""

import os
import select
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple, Callable
from threading import Thread
from enum import Enum

import cv2
import numpy as np

from core.utils import get_video_resolution_and_fps, get_best_stream_url, frame_to_ms
from core.source_router import SourceRouter

logger = logging.getLogger(__name__)


class StreamType(str, Enum):
    """Supported stream types"""
    HLS = "hls"
    RTSP = "rtsp"
    MP4 = "mp4"
    FILE = "file"


@dataclass
class FramePacket:
    """
    Container for a single frame and its metadata.

    This is the unit of data passed between frame provider and services.
    """
    frame_number: int
    frame: np.ndarray
    timestamp_ms: int
    width: int
    height: int

    # Optional metadata
    segment_number: Optional[int] = None  # For HLS streams

    def __post_init__(self):
        """Validate frame data"""
        if self.frame is None:
            raise ValueError("Frame cannot be None")
        if len(self.frame.shape) != 3:
            raise ValueError(f"Frame must be 3D array, got shape {self.frame.shape}")


@dataclass
class FrameProviderConfig:
    """Configuration for frame provider"""
    stream_url: str
    stream_type: StreamType = StreamType.HLS

    # Resolution settings
    target_width: Optional[int] = None  # None = use source resolution
    target_height: Optional[int] = None
    resolution_preference: str = "_1080p.m3u8"  # For HLS master playlists

    # Processing settings
    start_frame: int = 0  # Frame to start from (for resume)
    start_segment: int = 1  # HLS segment to start from
    frame_skip: int = 1  # Process every Nth frame

    # FFmpeg settings
    ffmpeg_timeout: float = 0.1  # Timeout for non-blocking read
    pixel_format: str = "bgr24"  # OpenCV compatible

    # Callbacks
    on_frame: Optional[Callable[[FramePacket], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
    on_complete: Optional[Callable[[], None]] = None


class FrameProvider:
    """
    Extracts frames from video streams using FFmpeg.

    This class handles:
    - HLS, RTSP, MP4, and local file streams
    - Automatic resolution detection and scaling
    - Non-blocking frame reading for live streams
    - Resume from specific segment/frame (for crash recovery)

    Usage:
        config = FrameProviderConfig(
            stream_url="https://example.com/stream.m3u8",
            stream_type=StreamType.HLS,
            target_width=1280,
            target_height=720
        )
        provider = FrameProvider(config)

        # Option 1: Iterator
        for frame_packet in provider.frames():
            process(frame_packet)

        # Option 2: Callback
        provider.start(on_frame=process_frame)
    """

    def __init__(self, config: FrameProviderConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[Thread] = None
        self._running = False

        # OpenCV capture (for local files)
        self._cv_capture: Optional[cv2.VideoCapture] = None

        # Stream metadata (populated on start)
        self.source_width: Optional[int] = None
        self.source_height: Optional[int] = None
        self.fps: Optional[float] = None
        self.effective_width: int = 0
        self.effective_height: int = 0

        # Frame counter
        self._frame_number = config.start_frame

    @property
    def _use_opencv(self) -> bool:
        """
        Determine if we should use OpenCV instead of FFmpeg.

        Delegates to SourceRouter - the single source of truth for type detection.

        Uses OpenCV for local files (faster, no subprocess overhead).
        Uses FFmpeg for streams (HLS, RTSP) which require special handling.
        """
        source_type = SourceRouter.detect(self.config.stream_url)
        return not SourceRouter.needs_ffmpeg(source_type)

    def _resolve_stream_url(self) -> str:
        """Resolve the actual stream URL (handle HLS master playlists)"""
        url = self.config.stream_url

        if self.config.stream_type == StreamType.HLS:
            # Try to get best quality variant
            url = get_best_stream_url(url, self.config.resolution_preference)
            logger.info(f"Resolved HLS URL: {url}")

        return url

    def _detect_stream_metadata(self, url: str) -> Tuple[int, int, float]:
        """Detect stream resolution and FPS"""
        width, height, fps = get_video_resolution_and_fps(url)

        if width is None or height is None or fps is None:
            raise RuntimeError(f"Failed to detect stream metadata for: {url}")

        logger.info(f"Stream metadata: {width}x{height} @ {fps:.2f} FPS")
        return width, height, fps

    def _build_ffmpeg_command(self, url: str) -> list:
        """Build FFmpeg command for frame extraction"""
        cmd = ["ffmpeg"]

        # Input options
        if self.config.stream_type == StreamType.HLS:
            # Start from specific segment for HLS (1-indexed in FFmpeg)
            cmd.extend(["-live_start_index", str(self.config.start_segment - 1)])

        # Input URL
        cmd.extend(["-i", url])

        # HLS specific options
        if self.config.stream_type == StreamType.HLS:
            cmd.extend(["-hls_flags", "delete_segments"])

        # Video filter for scaling
        if self.effective_width and self.effective_height:
            cmd.extend(["-vf", f"scale={self.effective_width}:{self.effective_height}"])

        # Output format (raw video to stdout)
        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", self.config.pixel_format,
            "-"
        ])

        return cmd

    def _start_ffmpeg(self, url: str) -> subprocess.Popen:
        """Start FFmpeg process"""
        cmd = self._build_ffmpeg_command(url)
        logger.info(f"Starting FFmpeg: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Start stderr logging thread
        self._stderr_thread = Thread(
            target=self._log_ffmpeg_stderr,
            args=(process,),
            daemon=True
        )
        self._stderr_thread.start()

        return process

    def _log_ffmpeg_stderr(self, process: subprocess.Popen):
        """Log FFmpeg stderr output (runs in separate thread)"""
        try:
            while process.poll() is None:
                if process.stderr:
                    error_output = process.stderr.read1(1024)
                    if error_output:
                        decoded = error_output.decode("utf-8", errors="ignore")
                        # Only log periodically to avoid spam
                        if self._frame_number % 1000 == 0:
                            logger.debug(f"FFmpeg: {decoded[:200]}")
        except Exception as e:
            logger.warning(f"Error reading FFmpeg stderr: {e}")

    def _read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from FFmpeg stdout.

        Uses select() for non-blocking read on live streams.
        """
        if not self._process or not self._process.stdout:
            return None

        frame_size = self.effective_width * self.effective_height * 3

        try:
            # Non-blocking wait for data
            rlist, _, _ = select.select(
                [self._process.stdout],
                [],
                [],
                self.config.ffmpeg_timeout
            )

            if not rlist:
                return None  # No data available yet

            # Read frame data
            raw_frame = self._process.stdout.read(frame_size)

            # Check for end of stream
            if len(raw_frame) != frame_size:
                logger.info("End of stream detected")
                return None

            # Convert to numpy array
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                (self.effective_height, self.effective_width, 3)
            )

            return frame

        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None

    # =========================================================================
    # OPENCV METHODS (for local files)
    # =========================================================================

    def _start_opencv(self) -> bool:
        """Start OpenCV video capture for local files"""
        try:
            self._cv_capture = cv2.VideoCapture(self.config.stream_url)

            if not self._cv_capture.isOpened():
                logger.error(f"Could not open video file: {self.config.stream_url}")
                return False

            # Seek to start_frame if specified
            if self.config.start_frame > 0:
                self._cv_capture.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)
                logger.info(f"Seeked to frame {self.config.start_frame}")

            logger.info(f"OpenCV capture started for: {self.config.stream_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to start OpenCV capture: {e}")
            return False

    def _read_frame_opencv(self) -> Optional[np.ndarray]:
        """Read a single frame using OpenCV"""
        if not self._cv_capture or not self._cv_capture.isOpened():
            return None

        try:
            ret, frame = self._cv_capture.read()

            if not ret:
                logger.info("End of video file detected")
                return None

            # Resize if needed
            if self.effective_width and self.effective_height:
                if frame.shape[1] != self.effective_width or frame.shape[0] != self.effective_height:
                    frame = cv2.resize(frame, (self.effective_width, self.effective_height))

            return frame

        except Exception as e:
            logger.error(f"Error reading frame with OpenCV: {e}")
            return None

    def _stop_opencv(self):
        """Release OpenCV capture"""
        if self._cv_capture:
            self._cv_capture.release()
            self._cv_capture = None
            logger.info("OpenCV capture released")

    def initialize(self) -> bool:
        """
        Initialize the frame provider.

        - Detects stream metadata from ORIGINAL URL (before resolution, like old code)
        - Resolves stream URL for actual playback
        - Sets up effective resolution

        Returns:
            True if initialization successful
        """
        try:
            # IMPORTANT: Detect metadata from ORIGINAL URL first (before resolving)
            # This matches old code behavior where fps is detected from master playlist
            self.source_width, self.source_height, self.fps = self._detect_stream_metadata(
                self.config.stream_url
            )

            # Determine effective resolution
            if self.config.target_width and self.config.target_height:
                self.effective_width = self.config.target_width
                self.effective_height = self.config.target_height
            else:
                self.effective_width = self.source_width
                self.effective_height = self.source_height

            logger.info(
                f"Initialized: source={self.source_width}x{self.source_height}, "
                f"effective={self.effective_width}x{self.effective_height}, "
                f"fps={self.fps}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize frame provider: {e}")
            if self.config.on_error:
                self.config.on_error(e)
            return False

    def frames(self) -> Iterator[FramePacket]:
        """
        Generator that yields FramePacket objects.

        Automatically uses OpenCV for local files (faster) or FFmpeg for streams.

        Usage:
            for frame_packet in provider.frames():
                process(frame_packet)
        """
        if not self.fps:
            if not self.initialize():
                return

        self._running = True
        self._frame_number = self.config.start_frame

        # Choose backend based on input type
        use_opencv = self._use_opencv

        if use_opencv:
            # Use OpenCV for local files
            logger.info(f"Using OpenCV backend for: {self.config.stream_url}")
            if not self._start_opencv():
                return
            yield from self._frames_opencv()
        else:
            # Use FFmpeg for streams
            logger.info(f"Using FFmpeg backend for: {self.config.stream_url}")
            url = self._resolve_stream_url()
            self._process = self._start_ffmpeg(url)
            yield from self._frames_ffmpeg()

    def _frames_ffmpeg(self) -> Iterator[FramePacket]:
        """Generate frames using FFmpeg backend"""
        try:
            while self._running:
                frame = self._read_frame()

                if frame is None:
                    # Check if stream ended or just no data yet
                    if self._process.poll() is not None:
                        # Process ended
                        logger.info("FFmpeg process ended")
                        break
                    continue

                # Apply frame skip
                if self.config.frame_skip > 1:
                    if self._frame_number % self.config.frame_skip != 0:
                        self._frame_number += 1
                        continue

                # Create frame packet
                packet = FramePacket(
                    frame_number=self._frame_number,
                    frame=frame,
                    timestamp_ms=frame_to_ms(self._frame_number, self.fps),
                    width=self.effective_width,
                    height=self.effective_height
                )

                yield packet

                # Log progress periodically
                if self._frame_number % 500 == 0:
                    logger.info(f"Processing frame {self._frame_number}")

                self._frame_number += 1

        except GeneratorExit:
            logger.info("Frame generator stopped by consumer")
        except Exception as e:
            logger.error(f"Error in FFmpeg frame generator: {e}")
            if self.config.on_error:
                self.config.on_error(e)
        finally:
            self.stop()
            if self.config.on_complete:
                self.config.on_complete()

    def _frames_opencv(self) -> Iterator[FramePacket]:
        """Generate frames using OpenCV backend"""
        try:
            while self._running:
                frame = self._read_frame_opencv()

                if frame is None:
                    # End of video
                    break

                # Apply frame skip
                if self.config.frame_skip > 1:
                    if self._frame_number % self.config.frame_skip != 0:
                        self._frame_number += 1
                        continue

                # Create frame packet
                packet = FramePacket(
                    frame_number=self._frame_number,
                    frame=frame,
                    timestamp_ms=frame_to_ms(self._frame_number, self.fps),
                    width=self.effective_width,
                    height=self.effective_height
                )

                yield packet

                # Log progress periodically
                if self._frame_number % 500 == 0:
                    logger.info(f"Processing frame {self._frame_number}")

                self._frame_number += 1

        except GeneratorExit:
            logger.info("Frame generator stopped by consumer")
        except Exception as e:
            logger.error(f"Error in OpenCV frame generator: {e}")
            if self.config.on_error:
                self.config.on_error(e)
        finally:
            self.stop()
            if self.config.on_complete:
                self.config.on_complete()

    def start(self, on_frame: Optional[Callable[[FramePacket], None]] = None):
        """
        Start frame extraction with callback.

        Args:
            on_frame: Callback function for each frame
        """
        callback = on_frame or self.config.on_frame

        if not callback:
            raise ValueError("No frame callback provided")

        for packet in self.frames():
            callback(packet)

    def stop(self):
        """Stop frame extraction and cleanup"""
        self._running = False

        # Stop FFmpeg if running
        if self._process:
            try:
                if self._process.stdout:
                    self._process.stdout.close()
                if self._process.stderr:
                    self._process.stderr.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping FFmpeg: {e}")
            finally:
                self._process = None

        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=2)

        # Stop OpenCV if running
        self._stop_opencv()

        logger.info("Frame provider stopped")

    @property
    def frame_count(self) -> int:
        """Current frame count"""
        return self._frame_number

    @property
    def is_running(self) -> bool:
        """Check if provider is running"""
        return self._running

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_frame_provider(
    stream_url: str,
    stream_type: str = "hls",
    width: Optional[int] = None,
    height: Optional[int] = None,
    start_frame: int = 0,
    frame_skip: int = 1
) -> FrameProvider:
    """
    Factory function to create a configured FrameProvider.

    Args:
        stream_url: URL or path to video source
        stream_type: Type of stream (hls, rtsp, mp4, file)
        width: Target width (None = source resolution)
        height: Target height (None = source resolution)
        start_frame: Frame to start from
        frame_skip: Process every Nth frame

    Returns:
        Configured FrameProvider instance
    """
    config = FrameProviderConfig(
        stream_url=stream_url,
        stream_type=StreamType(stream_type),
        target_width=width,
        target_height=height,
        start_frame=start_frame,
        frame_skip=frame_skip
    )

    return FrameProvider(config)


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Test frame provider")
    parser.add_argument("url", help="Stream URL or file path")
    parser.add_argument("--type", default="hls", choices=["hls", "rtsp", "mp4", "file"])
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=100)

    args = parser.parse_args()

    provider = create_frame_provider(
        stream_url=args.url,
        stream_type=args.type,
        width=args.width,
        height=args.height
    )

    frame_count = 0
    for packet in provider.frames():
        print(f"Frame {packet.frame_number}: {packet.width}x{packet.height} @ {packet.timestamp_ms}ms")
        frame_count += 1
        if frame_count >= args.max_frames:
            break

    print(f"Processed {frame_count} frames")
