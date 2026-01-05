"""
Core Utilities

Common utility functions extracted and cleaned from old inference code.
"""

import os
import subprocess
import shlex
import logging
from typing import Tuple, Optional, List
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# TIME/FRAME UTILITIES
# =============================================================================

def frame_to_timecode(frame_idx: int, fps: float, format: str = "HH:MM:SS.mmm") -> str:
    """
    Convert frame index to timecode string.

    Args:
        frame_idx: Frame number (0-indexed)
        fps: Frames per second
        format: Output format
            - "HH:MM:SS.mmm" -> "01:23:45.678"
            - "HH:MM:SS" -> "01:23:45"
            - "Seconds" -> "5025.678"
            - "HH_MM_SS" -> "01_23_45"
            - "ms" -> Milliseconds as integer

    Returns:
        Formatted timecode string
    """
    try:
        if fps <= 0:
            return "00:00:00.000"

        total_seconds = frame_idx / fps
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = total_seconds % 60

        if format == "HH:MM:SS.mmm":
            return f"{h:02d}:{m:02d}:{s:06.3f}"
        elif format == "HH:MM:SS":
            return f"{h:02d}:{m:02d}:{int(s):02d}"
        elif format == "Seconds":
            return f"{total_seconds:.3f}"
        elif format == "HH_MM_SS":
            return f"{h:02d}_{m:02d}_{int(s):02d}"
        elif format == "ms":
            return str(int(total_seconds * 1000))
        else:
            return f"{h:02d}:{m:02d}:{s:06.3f}"

    except Exception as e:
        logger.error(f"Error in frame_to_timecode: {e}")
        return "00:00:00.000"


def frame_to_ms(frame_idx: int, fps: float) -> int:
    """Convert frame index to milliseconds."""
    if fps <= 0:
        return 0
    return int((frame_idx / fps) * 1000)


def ms_to_frame(ms: int, fps: float) -> int:
    """Convert milliseconds to frame index."""
    if fps <= 0:
        return 0
    return int((ms / 1000) * fps)


# =============================================================================
# VIDEO/STREAM UTILITIES
# =============================================================================

def get_video_resolution_and_fps(
    source_path: str,
    use_avg_frame_rate: bool = False
) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """
    Get video width, height, and FPS from a video file or stream URL.

    Uses ffprobe to extract stream metadata.

    Args:
        source_path: Local file path or video stream URL (.mp4 or .m3u8)
        use_avg_frame_rate: If True, use avg_frame_rate (actual delivered rate).
                           If False, use r_frame_rate (container rate, like old code).
                           Default False for backward compatibility.

    Returns:
        Tuple of (width, height, fps) or (None, None, None) on error

    Note:
        r_frame_rate: The "real" frame rate from container metadata (may be inaccurate for HLS)
        avg_frame_rate: The average frame rate computed from stream data (more accurate)

        For HLS streams, r_frame_rate often reports 60fps while actual content is 24/25fps.
        Use avg_frame_rate=True for more accurate frame rate detection.
    """
    try:
        # Get both frame rates and let caller decide which to use
        frame_rate_field = "avg_frame_rate" if use_avg_frame_rate else "r_frame_rate"

        cmd = (
            f'ffprobe -v error -select_streams v:0 '
            f'-show_entries stream=width,height,{frame_rate_field} '
            f'-of default=noprint_wrappers=1 "{source_path}"'
        )
        output = subprocess.check_output(
            shlex.split(cmd),
            timeout=30
        ).decode().strip().split()

        width = int(output[0].split('=')[1])
        height = int(output[1].split('=')[1])
        fps_str = output[2].split('=')[1]  # e.g. "30000/1001"

        if '/' in fps_str:
            num, denom = map(int, fps_str.split('/'))
            fps = round(num / denom, 6) if denom != 0 else 0.0
        else:
            fps = float(fps_str)

        return width, height, fps

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout getting video info for: {source_path}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error getting resolution and FPS for {source_path}: {e}")
        return None, None, None


def get_best_stream_url(m3u8_url: str, resolution_suffix: str = "_1080p.m3u8") -> str:
    """
    Get the best quality stream URL from an HLS master playlist.

    Args:
        m3u8_url: URL to the master playlist
        resolution_suffix: Preferred resolution suffix (e.g., "_1080p.m3u8")

    Returns:
        URL to the best matching stream, or original URL if not found
    """
    try:
        resp = requests.get(m3u8_url, timeout=10)
        resp.raise_for_status()
        content = resp.text.strip().splitlines()

        # Look for preferred resolution entry in playlist
        for line in content:
            if line.endswith(resolution_suffix):
                return urljoin(m3u8_url, line.strip())

        # Fallback to original URL
        return m3u8_url

    except Exception as e:
        logger.warning(f"Error getting best stream URL: {e}")
        return m3u8_url


# =============================================================================
# FILE/DOWNLOAD UTILITIES
# =============================================================================

def download_file(url: str, local_path: str, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL to local path.

    Args:
        url: Source URL
        local_path: Destination file path
        chunk_size: Download chunk size in bytes

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        logger.info(f"Downloaded: {url} -> {local_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def ensure_model_available(
    model_url: str,
    local_path: str,
    force_download: bool = False
) -> str:
    """
    Ensure model file is available locally, downloading if needed.

    Args:
        model_url: URL to download model from
        local_path: Local path to store/check model
        force_download: If True, download even if file exists

    Returns:
        Path to the local model file

    Raises:
        RuntimeError: If model cannot be obtained
    """
    if os.path.exists(local_path) and not force_download:
        logger.info(f"Model already exists: {local_path}")
        return local_path

    if not download_file(model_url, local_path):
        raise RuntimeError(f"Failed to download model from {model_url}")

    return local_path


# =============================================================================
# BBOX UTILITIES
# =============================================================================

def normalize_bbox(
    bbox: List[float],
    width: int,
    height: int
) -> List[float]:
    """
    Normalize bounding box coordinates to 0-1 range.

    Args:
        bbox: [x1, y1, x2, y2] in pixel coordinates
        width: Image width
        height: Image height

    Returns:
        [x1, y1, x2, y2] normalized to 0-1 range
    """
    x1, y1, x2, y2 = bbox
    return [
        round(x1 / width, 6),
        round(y1 / height, 6),
        round(x2 / width, 6),
        round(y2 / height, 6)
    ]


def denormalize_bbox(
    bbox: List[float],
    width: int,
    height: int
) -> List[int]:
    """
    Convert normalized bounding box back to pixel coordinates.

    Args:
        bbox: [x1, y1, x2, y2] normalized (0-1 range)
        width: Image width
        height: Image height

    Returns:
        [x1, y1, x2, y2] in pixel coordinates
    """
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height)
    ]


def clip_bbox(bbox: List[float], max_val: float = 1.0) -> List[float]:
    """Clip bbox coordinates to valid range."""
    return [max(0.0, min(max_val, v)) for v in bbox]


def bbox_area(bbox: List[float]) -> float:
    """Calculate bbox area (works for both normalized and pixel coords)."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union between two bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
