"""
Camera View Service

Detects camera cuts/view changes using perceptual hashing and histogram comparison.
Extracted and refactored from old_inference_code_for_reference/src/live_stream_processing/camera_view_detection.py
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image
import imagehash

from services.base_service import BaseService, ServiceConfig
from core.frame_provider import FramePacket

logger = logging.getLogger(__name__)


@dataclass
class CameraViewServiceConfig(ServiceConfig):
    """
    Configuration for Camera View Service.
    """
    # Detection thresholds
    phash_threshold: int = 20  # Hamming distance threshold for pHash
    histogram_threshold: float = 0.90  # Correlation threshold (below = cut)
    min_frame_gap: int = 0  # Minimum frames between detected cuts (0 = auto from fps)

    # Processing settings
    resolution_scale: float = 0.5  # Scale factor for processing (saves compute)

    @classmethod
    def from_game_config(
        cls,
        match_id: str,
        stream_url: str,
        service_config: Dict[str, Any],
        inference_settings: Dict[str, Any]
    ) -> "CameraViewServiceConfig":
        """
        Create CameraViewServiceConfig from GameConfig components.
        """
        params = service_config.get("params", {})
        res = inference_settings.get("processing_resolution", [1280, 720])

        return cls(
            match_id=match_id,
            service_id="camera_view",
            stream_url=stream_url,
            stream_type="hls",
            # Use scaled resolution for camera view (less compute needed)
            target_width=int(res[0] * params.get("resolution_scale", 0.5)),
            target_height=int(res[1] * params.get("resolution_scale", 0.5)),
            frame_skip=inference_settings.get("frame_skip", 1),
            device="cpu",  # Camera view runs on CPU
            phash_threshold=params.get("phash_threshold", 20),
            histogram_threshold=params.get("histogram_threshold", 0.90),
            min_frame_gap=params.get("min_frame_gap", 25),
            resolution_scale=params.get("resolution_scale", 0.5),
        )


class CameraViewService(BaseService):
    """
    Camera View Detection Service.

    Uses perceptual hashing (pHash) and color histogram comparison
    to detect camera cuts/view changes in video streams.

    Algorithm:
    1. Compute perceptual hash of each frame
    2. Compute color histogram (HSV) of each frame
    3. Compare with previous frame:
       - If pHash difference >= threshold AND histogram correlation < threshold
       - AND minimum frame gap has passed
       - Then: camera cut detected

    Usage:
        config = CameraViewServiceConfig(
            match_id="match_123",
            stream_url="https://example.com/stream.m3u8",
            phash_threshold=20,
            histogram_threshold=0.90,
        )
        service = CameraViewService(config)
        service.run()
    """

    def __init__(self, config: CameraViewServiceConfig):
        super().__init__(config)
        self.cv_config = config

        # State for comparison
        self._prev_phash: Optional[imagehash.ImageHash] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._last_cut_frame: int = -1000  # Large negative to allow first detection
        self._effective_min_frame_gap: int = 25  # Default, will be computed from fps in initialize()

    def initialize(self) -> bool:
        """
        Initialize Camera View service.

        Returns:
            True if initialization successful
        """
        try:
            # Verify imagehash is available
            import imagehash

            # Compute min_frame_gap from fps if set to auto (0)
            # Old code: MIN_FRAME_DIFF = int(fps) -> 1 second worth of frames
            if self.cv_config.min_frame_gap == 0 and self._frame_provider and self._frame_provider.fps:
                self._effective_min_frame_gap = int(self._frame_provider.fps)
                logger.info(f"Auto min_frame_gap from fps: {self._effective_min_frame_gap}")
            else:
                self._effective_min_frame_gap = self.cv_config.min_frame_gap if self.cv_config.min_frame_gap > 0 else 25

            logger.info(
                f"Camera View Service initialized: "
                f"phash_threshold={self.cv_config.phash_threshold}, "
                f"histogram_threshold={self.cv_config.histogram_threshold}, "
                f"min_frame_gap={self._effective_min_frame_gap}"
            )
            return True

        except ImportError:
            logger.error("imagehash package is required. Install with: pip install imagehash")
            return False

    def _compute_frame_features(
        self,
        frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[imagehash.ImageHash]]:
        """
        Compute perceptual hash and color histogram for a frame.

        Args:
            frame_bgr: Frame in BGR format

        Returns:
            Tuple of (histogram, phash) or (None, None) on error
        """
        try:
            # Compute color histogram (HSV with 50x50x50 bins)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist)

            # Compute perceptual hash
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            phash = imagehash.phash(pil_image)

            return hist, phash

        except Exception as e:
            logger.error(f"Error computing frame features: {e}")
            return None, None

    def _detect_camera_cut(
        self,
        hist: np.ndarray,
        phash: imagehash.ImageHash,
        frame_number: int
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        """
        Detect if this frame represents a camera cut.

        Args:
            hist: Current frame histogram
            phash: Current frame perceptual hash
            frame_number: Current frame number

        Returns:
            Tuple of (is_cut, phash_diff, histogram_correlation)
        """
        # First frame - no comparison possible
        if self._prev_phash is None or self._prev_hist is None:
            self._prev_phash = phash
            self._prev_hist = hist
            return False, None, None

        # Compute similarities
        phash_diff = self._prev_phash - phash  # Hamming distance
        hist_corr = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)

        # Update previous values
        self._prev_phash = phash
        self._prev_hist = hist

        # Check if minimum frame gap has passed
        frames_since_last = frame_number - self._last_cut_frame
        if frames_since_last < self._effective_min_frame_gap:
            return False, phash_diff, hist_corr

        # Detect cut: high pHash difference AND low histogram correlation
        is_cut = (
            phash_diff >= self.cv_config.phash_threshold and
            (hist_corr < self.cv_config.histogram_threshold or np.isnan(hist_corr))
        )

        if is_cut:
            self._last_cut_frame = frame_number
            logger.info(
                f"Camera cut detected at frame {frame_number}: "
                f"phash_diff={phash_diff}, hist_corr={hist_corr:.3f}"
            )

        return is_cut, phash_diff, hist_corr

    def process_frame(self, frame_packet: FramePacket) -> Optional[Dict[str, Any]]:
        """
        Process a single frame - detect camera cut.

        Args:
            frame_packet: Frame data and metadata

        Returns:
            Dict with camera cut info (only if cut detected), else None
        """
        try:
            # Compute features
            hist, phash = self._compute_frame_features(frame_packet.frame)

            if hist is None or phash is None:
                return None

            # Detect camera cut
            is_cut, phash_diff, hist_corr = self._detect_camera_cut(
                hist, phash, frame_packet.frame_number
            )

            # Only write to DB if camera cut detected
            if is_cut:
                return {
                    "is_camera_cut": True,
                    "phash_diff": int(phash_diff) if phash_diff is not None else None,
                    "histogram_correlation": round(float(hist_corr), 4) if hist_corr is not None else None,
                }

            # Return None to skip DB write for non-cut frames
            return None

        except Exception as e:
            logger.error(f"Error processing frame {frame_packet.frame_number}: {e}")
            return None

    def cleanup(self):
        """Clean up service resources"""
        self._prev_phash = None
        self._prev_hist = None
        logger.info("Camera View Service cleaned up")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for running Camera View service standalone"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Camera View Detection Service")
    parser.add_argument("--match-id", required=True, help="Match identifier")
    parser.add_argument("--stream-url", required=True, help="Stream URL")
    parser.add_argument("--phash-threshold", type=int, default=20, help="pHash difference threshold")
    parser.add_argument("--histogram-threshold", type=float, default=0.90, help="Histogram correlation threshold")
    parser.add_argument("--min-frame-gap", type=int, default=25, help="Minimum frames between cuts")
    parser.add_argument("--width", type=int, help="Processing width")
    parser.add_argument("--height", type=int, help="Processing height")
    parser.add_argument("--db-table", default="inference_results", help="DynamoDB table name")
    parser.add_argument("--local-output", help="Local output directory (for testing without DynamoDB)")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame number to start from (for resume)")
    parser.add_argument("--start-segment", type=int, default=1, help="Segment number to start from (for HLS resume)")

    args = parser.parse_args()

    # Create config
    config = CameraViewServiceConfig(
        match_id=args.match_id,
        service_id="camera_view",
        stream_url=args.stream_url,
        target_width=args.width,
        target_height=args.height,
        phash_threshold=args.phash_threshold,
        histogram_threshold=args.histogram_threshold,
        min_frame_gap=args.min_frame_gap,
        db_table_name=args.db_table,
        local_output_dir=args.local_output,
        start_frame=args.start_frame,
        start_segment=args.start_segment,
    )

    # Run service
    service = CameraViewService(config)
    service.run()


if __name__ == "__main__":
    main()
