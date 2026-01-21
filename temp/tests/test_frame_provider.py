"""
Tests for Frame Provider

Validates both OpenCV (local files) and FFmpeg (streams) backends work correctly.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.frame_provider import (
    FrameProvider,
    FrameProviderConfig,
    FramePacket,
    StreamType,
    create_frame_provider
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_mp4_path(tmp_path):
    """Create a sample MP4 file for testing"""
    video_path = tmp_path / "test_video.mp4"

    # Create a simple test video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    frame_count = 100
    width, height = 640, 480

    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(frame_count):
        # Create a frame with varying color to detect frame differences
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 2 % 256  # Blue channel varies
        frame[:, :, 1] = 100  # Green constant
        frame[:, :, 2] = 50  # Red constant

        # Add frame number text
        cv2.putText(frame, f"Frame {i}", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        out.write(frame)

    out.release()
    return str(video_path)


# =============================================================================
# BACKEND SELECTION TESTS
# =============================================================================

class TestBackendSelection:
    """Test that correct backend is selected based on input type"""

    def test_mp4_stream_type_uses_opencv(self):
        """StreamType.MP4 should use OpenCV"""
        config = FrameProviderConfig(
            stream_url="/path/to/video.mp4",
            stream_type=StreamType.MP4
        )
        provider = FrameProvider(config)
        assert provider._use_opencv is True

    def test_file_stream_type_uses_opencv(self):
        """StreamType.FILE should use OpenCV"""
        config = FrameProviderConfig(
            stream_url="/path/to/video.avi",
            stream_type=StreamType.FILE
        )
        provider = FrameProvider(config)
        assert provider._use_opencv is True

    def test_hls_stream_type_uses_ffmpeg(self):
        """StreamType.HLS should use FFmpeg"""
        config = FrameProviderConfig(
            stream_url="https://example.com/stream.m3u8",
            stream_type=StreamType.HLS
        )
        provider = FrameProvider(config)
        assert provider._use_opencv is False

    def test_rtsp_stream_type_uses_ffmpeg(self):
        """StreamType.RTSP should use FFmpeg"""
        config = FrameProviderConfig(
            stream_url="rtsp://example.com/stream",
            stream_type=StreamType.RTSP
        )
        provider = FrameProvider(config)
        assert provider._use_opencv is False

    def test_local_file_path_uses_opencv(self, sample_mp4_path):
        """Local file path should use OpenCV even with HLS stream type"""
        config = FrameProviderConfig(
            stream_url=sample_mp4_path,
            stream_type=StreamType.HLS  # Wrong type, but local file detected
        )
        provider = FrameProvider(config)
        assert provider._use_opencv is True

    def test_mp4_extension_without_http_uses_opencv(self):
        """Local .mp4 path should use OpenCV"""
        config = FrameProviderConfig(
            stream_url="/videos/match.mp4",
            stream_type=StreamType.HLS
        )
        provider = FrameProvider(config)
        assert provider._use_opencv is True

    def test_http_mp4_uses_ffmpeg(self):
        """HTTP URL to MP4 should use FFmpeg (not OpenCV)"""
        config = FrameProviderConfig(
            stream_url="https://example.com/video.mp4",
            stream_type=StreamType.HLS
        )
        provider = FrameProvider(config)
        # HTTP URLs should NOT use OpenCV even with video extension
        assert provider._use_opencv is False


# =============================================================================
# OPENCV BACKEND TESTS
# =============================================================================

class TestOpenCVBackend:
    """Test OpenCV backend for local files"""

    def test_opencv_reads_frames(self, sample_mp4_path):
        """OpenCV backend should read frames from MP4"""
        config = FrameProviderConfig(
            stream_url=sample_mp4_path,
            stream_type=StreamType.MP4
        )
        provider = FrameProvider(config)

        frames = []
        for packet in provider.frames():
            frames.append(packet)
            if len(frames) >= 10:
                break

        assert len(frames) == 10
        assert all(isinstance(f, FramePacket) for f in frames)

    def test_opencv_frame_packet_structure(self, sample_mp4_path):
        """Frame packets should have correct structure"""
        config = FrameProviderConfig(
            stream_url=sample_mp4_path,
            stream_type=StreamType.MP4
        )
        provider = FrameProvider(config)

        for packet in provider.frames():
            assert packet.frame_number >= 0
            assert isinstance(packet.frame, np.ndarray)
            assert packet.frame.shape == (480, 640, 3)  # height, width, channels
            assert packet.width == 640
            assert packet.height == 480
            assert packet.timestamp_ms >= 0
            break

    def test_opencv_frame_skip(self, sample_mp4_path):
        """Frame skip should work correctly"""
        config = FrameProviderConfig(
            stream_url=sample_mp4_path,
            stream_type=StreamType.MP4,
            frame_skip=3
        )
        provider = FrameProvider(config)

        frame_numbers = []
        for packet in provider.frames():
            frame_numbers.append(packet.frame_number)
            if len(frame_numbers) >= 5:
                break

        # With frame_skip=3, we should get frames 0, 3, 6, 9, 12
        assert frame_numbers == [0, 3, 6, 9, 12]

    def test_opencv_start_frame(self, sample_mp4_path):
        """Start frame should seek correctly"""
        config = FrameProviderConfig(
            stream_url=sample_mp4_path,
            stream_type=StreamType.MP4,
            start_frame=50
        )
        provider = FrameProvider(config)

        for packet in provider.frames():
            assert packet.frame_number == 50
            break

    def test_opencv_resize(self, sample_mp4_path):
        """Target resolution should resize frames"""
        config = FrameProviderConfig(
            stream_url=sample_mp4_path,
            stream_type=StreamType.MP4,
            target_width=320,
            target_height=240
        )
        provider = FrameProvider(config)

        for packet in provider.frames():
            assert packet.width == 320
            assert packet.height == 240
            assert packet.frame.shape == (240, 320, 3)
            break

    def test_opencv_metadata_detection(self, sample_mp4_path):
        """Should detect video metadata correctly"""
        config = FrameProviderConfig(
            stream_url=sample_mp4_path,
            stream_type=StreamType.MP4
        )
        provider = FrameProvider(config)
        provider.initialize()

        assert provider.source_width == 640
        assert provider.source_height == 480
        assert provider.fps == pytest.approx(30.0, rel=0.1)


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunction:
    """Test the create_frame_provider factory function"""

    def test_create_frame_provider_basic(self, sample_mp4_path):
        """Factory should create working provider"""
        provider = create_frame_provider(
            stream_url=sample_mp4_path,
            stream_type="mp4"
        )

        assert isinstance(provider, FrameProvider)
        assert provider.config.stream_type == StreamType.MP4

    def test_create_frame_provider_with_options(self, sample_mp4_path):
        """Factory should accept all options"""
        provider = create_frame_provider(
            stream_url=sample_mp4_path,
            stream_type="file",
            width=320,
            height=240,
            start_frame=10,
            frame_skip=2
        )

        assert provider.config.target_width == 320
        assert provider.config.target_height == 240
        assert provider.config.start_frame == 10
        assert provider.config.frame_skip == 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for frame provider"""

    def test_full_video_processing(self, sample_mp4_path):
        """Process entire video file"""
        config = FrameProviderConfig(
            stream_url=sample_mp4_path,
            stream_type=StreamType.MP4
        )
        provider = FrameProvider(config)

        frame_count = 0
        for packet in provider.frames():
            frame_count += 1

        # Should process all 100 frames
        assert frame_count == 100

    def test_context_manager(self, sample_mp4_path):
        """Context manager should work correctly"""
        config = FrameProviderConfig(
            stream_url=sample_mp4_path,
            stream_type=StreamType.MP4
        )

        with FrameProvider(config) as provider:
            assert provider.fps is not None
            frames = list(provider.frames())
            assert len(frames) > 0

    def test_stop_early(self, sample_mp4_path):
        """Should handle early stop gracefully"""
        config = FrameProviderConfig(
            stream_url=sample_mp4_path,
            stream_type=StreamType.MP4
        )
        provider = FrameProvider(config)

        frame_count = 0
        for packet in provider.frames():
            frame_count += 1
            if frame_count >= 20:
                provider.stop()
                break

        assert frame_count == 20
        assert not provider.is_running


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
