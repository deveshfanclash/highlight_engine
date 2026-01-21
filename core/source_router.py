"""
Source Router

Single point of truth for detecting and routing input source types.
All source type detection logic lives HERE and nowhere else.

This follows the Router/Factory Isolation pattern:
- ALL "what type is this?" logic in ONE place
- Rest of system NEVER checks types directly
"""

import os
from enum import Enum
from typing import Optional
from pathlib import Path


class SourceType(str, Enum):
    """
    Detected source types.

    More granular than InputType - captures the actual nature of the source.
    """
    HLS_STREAM = "hls"
    RTSP_STREAM = "rtsp"
    LOCAL_VIDEO = "local_video"
    LOCAL_IMAGE = "local_image"
    REMOTE_VIDEO = "remote_video"


class SourceRouter:
    """
    Single place for ALL source type detection.

    Usage:
        source_type = SourceRouter.detect("/path/to/video.mp4")
        # Returns: SourceType.LOCAL_VIDEO

        source_type = SourceRouter.detect("https://cdn.example.com/stream.m3u8")
        # Returns: SourceType.HLS_STREAM

        # Check properties
        if SourceRouter.needs_ffmpeg(source_type):
            # Use FFmpeg backend
        else:
            # Use OpenCV backend

    Why this exists:
        Before: Type detection scattered in main.py, frame_provider.py
        After: ONE place makes the decision, rest of system uses the result
    """

    # Video file extensions
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

    # Image file extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

    # Stream protocols
    STREAM_PROTOCOLS = {'rtsp://', 'rtmp://'}

    @classmethod
    def detect(cls, source: str) -> SourceType:
        """
        Detect source type from URL or path.

        This is the ONLY place type detection happens in the entire codebase.

        Args:
            source: URL or file path

        Returns:
            SourceType enum value
        """
        if not source:
            raise ValueError("Source cannot be empty")

        lower = source.lower().strip()

        # 1. Check for RTSP/RTMP streams (protocol-based)
        for protocol in cls.STREAM_PROTOCOLS:
            if lower.startswith(protocol):
                return SourceType.RTSP_STREAM

        # 2. Check for HLS streams (.m3u8)
        if '.m3u8' in lower:
            return SourceType.HLS_STREAM

        # 3. Check if it's a local file that exists
        if cls._is_local_file(source):
            return cls._detect_local_file_type(source)

        # 4. Check for remote URLs
        if lower.startswith(('http://', 'https://')):
            # Could be HLS without .m3u8 in URL, or direct video
            # Default to remote video - can be refined if needed
            return SourceType.REMOTE_VIDEO

        # 5. Fallback: assume it's a local video path (might not exist yet)
        ext = Path(source).suffix.lower()
        if ext in cls.IMAGE_EXTENSIONS:
            return SourceType.LOCAL_IMAGE

        return SourceType.LOCAL_VIDEO

    @classmethod
    def _is_local_file(cls, source: str) -> bool:
        """Check if source is a local file that exists."""
        # Don't treat URLs as local files
        if source.lower().startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
            return False
        return os.path.isfile(source)

    @classmethod
    def _detect_local_file_type(cls, source: str) -> SourceType:
        """Detect type of a local file."""
        ext = Path(source).suffix.lower()

        if ext in cls.IMAGE_EXTENSIONS:
            return SourceType.LOCAL_IMAGE

        # Default to video for local files
        return SourceType.LOCAL_VIDEO

    @classmethod
    def needs_ffmpeg(cls, source_type: SourceType) -> bool:
        """
        Does this source type require FFmpeg for reading?

        FFmpeg is needed for:
        - HLS streams (segment fetching, decryption)
        - RTSP streams (network protocol handling)

        OpenCV is preferred for:
        - Local files (faster, no subprocess)
        - Direct video URLs (simpler)
        """
        return source_type in (SourceType.HLS_STREAM, SourceType.RTSP_STREAM)

    @classmethod
    def is_stream(cls, source_type: SourceType) -> bool:
        """Is this a live/continuous stream (vs finite file)?"""
        return source_type in (SourceType.HLS_STREAM, SourceType.RTSP_STREAM)

    @classmethod
    def is_local(cls, source_type: SourceType) -> bool:
        """Is this a local file?"""
        return source_type in (SourceType.LOCAL_VIDEO, SourceType.LOCAL_IMAGE)

    @classmethod
    def is_video(cls, source_type: SourceType) -> bool:
        """Is this a video source (vs image)?"""
        return source_type != SourceType.LOCAL_IMAGE

    @classmethod
    def to_input_type(cls, source_type: SourceType) -> "InputType":
        """
        Convert SourceType to existing InputType enum.

        This bridges the new SourceRouter to the existing config schema.
        """
        from config.schemas import InputType

        mapping = {
            SourceType.HLS_STREAM: InputType.HLS,
            SourceType.RTSP_STREAM: InputType.RTSP,
            SourceType.LOCAL_VIDEO: InputType.MP4,
            SourceType.LOCAL_IMAGE: InputType.FILE,
            SourceType.REMOTE_VIDEO: InputType.HLS,  # Treat as HLS for FFmpeg handling
        }
        return mapping.get(source_type, InputType.FILE)

    @classmethod
    def to_stream_type(cls, source_type: SourceType) -> "StreamType":
        """
        Convert SourceType to FrameProvider's StreamType enum.

        This bridges the new SourceRouter to the frame provider.
        """
        from core.frame_provider import StreamType

        mapping = {
            SourceType.HLS_STREAM: StreamType.HLS,
            SourceType.RTSP_STREAM: StreamType.RTSP,
            SourceType.LOCAL_VIDEO: StreamType.MP4,
            SourceType.LOCAL_IMAGE: StreamType.FILE,
            SourceType.REMOTE_VIDEO: StreamType.HLS,
        }
        return mapping.get(source_type, StreamType.FILE)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_source_type(source: str) -> SourceType:
    """Convenience function for SourceRouter.detect()"""
    return SourceRouter.detect(source)


def needs_ffmpeg(source: str) -> bool:
    """Convenience function to check if source needs FFmpeg."""
    source_type = SourceRouter.detect(source)
    return SourceRouter.needs_ffmpeg(source_type)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test cases
    test_sources = [
        # HLS streams
        ("https://cdn.example.com/stream.m3u8", SourceType.HLS_STREAM),
        ("https://cdn.example.com/live/playlist.m3u8?token=abc", SourceType.HLS_STREAM),

        # RTSP streams
        ("rtsp://192.168.1.100:554/stream", SourceType.RTSP_STREAM),
        ("rtmp://live.example.com/app/stream", SourceType.RTSP_STREAM),

        # Remote videos
        ("https://example.com/video.mp4", SourceType.REMOTE_VIDEO),

        # Local videos (paths that don't exist - detected by extension)
        ("/path/to/video.mp4", SourceType.LOCAL_VIDEO),
        ("./test.avi", SourceType.LOCAL_VIDEO),

        # Local images
        ("/path/to/image.jpg", SourceType.LOCAL_IMAGE),
        ("./frame.png", SourceType.LOCAL_IMAGE),
    ]

    print("Testing SourceRouter...")
    for source, expected in test_sources:
        result = SourceRouter.detect(source)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {source[:50]:50} -> {result.value:15} (expected: {expected.value})")

    print("\nTesting properties...")
    print(f"  HLS needs FFmpeg: {SourceRouter.needs_ffmpeg(SourceType.HLS_STREAM)}")
    print(f"  Local video needs FFmpeg: {SourceRouter.needs_ffmpeg(SourceType.LOCAL_VIDEO)}")
    print(f"  HLS is stream: {SourceRouter.is_stream(SourceType.HLS_STREAM)}")
    print(f"  Local video is stream: {SourceRouter.is_stream(SourceType.LOCAL_VIDEO)}")
