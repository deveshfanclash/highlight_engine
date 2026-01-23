"""
Model Downloader

Downloads model weights from various sources (always fresh, no caching):
- Local files (returned as-is)
- HTTP/HTTPS URLs
- HuggingFace Hub
- S3

Usage:
    # Download from URL
    path = download_model("https://example.com/model.pt")

    # Download from S3
    path = download_model("s3://bucket/models/yolo.pt")

    # Local file (just validates and returns)
    path = download_model("/path/to/model.pt")
"""

import os
import tempfile
import logging
from pathlib import Path
from enum import Enum
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Model source types."""
    LOCAL = "local"
    URL = "url"
    HUGGINGFACE = "huggingface"
    S3 = "s3"


# Temp directory for downloaded models (cleaned up on process exit)
_download_dir: Path = None


def _get_download_dir() -> Path:
    """Get or create temp directory for downloads."""
    global _download_dir
    if _download_dir is None or not _download_dir.exists():
        _download_dir = Path(tempfile.mkdtemp(prefix="infer_models_"))
        logger.info(f"Model download directory: {_download_dir}")
    return _download_dir


def detect_source_type(source: str) -> SourceType:
    """Detect the type of model source."""
    if os.path.exists(source):
        return SourceType.LOCAL

    if source.startswith("s3://"):
        return SourceType.S3

    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        if "huggingface.co" in parsed.netloc:
            return SourceType.HUGGINGFACE
        return SourceType.URL

    # Check if it looks like a HuggingFace model ID (org/model)
    if "/" in source and not source.startswith("/"):
        parts = source.split("/")
        if len(parts) == 2 and all(p.replace("-", "").replace("_", "").isalnum() for p in parts):
            return SourceType.HUGGINGFACE

    # Assume local path
    return SourceType.LOCAL


def _download_url(url: str, dest_path: Path, timeout: int = 300) -> Path:
    """Download from HTTP/HTTPS URL."""
    logger.info(f"Downloading: {url}")

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Log progress every 10MB
                    if total_size > 0 and downloaded % (10 * 1024 * 1024) < 8192:
                        pct = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {pct:.1f}%")

    logger.info(f"Downloaded to: {dest_path}")
    return dest_path


def _download_huggingface(source: str, dest_path: Path) -> Path:
    """Download from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        # Fall back to URL if huggingface_hub not installed
        logger.warning("huggingface_hub not installed, trying URL fallback")

        if source.startswith("https://"):
            return _download_url(source, dest_path)

        # Convert model ID to URL
        parts = source.split("/")
        if len(parts) >= 2:
            filename = dest_path.name
            url = f"https://huggingface.co/{source}/resolve/main/{filename}"
            return _download_url(url, dest_path)

        raise RuntimeError(f"Cannot parse HuggingFace model ID: {source}")

    # Parse source: could be "org/model" or "org/model/filename.pt"
    parts = source.split("/")
    if len(parts) == 2:
        repo_id = source
        filename = dest_path.name
    elif len(parts) >= 3:
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
    else:
        raise RuntimeError(f"Invalid HuggingFace source: {source}")

    logger.info(f"Downloading from HuggingFace: {repo_id}/{filename}")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_path.parent,
        local_dir_use_symlinks=False,
        force_download=True,  # Always fresh
    )

    return Path(downloaded_path)


def _download_s3(source: str, dest_path: Path) -> Path:
    """Download from S3."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 not installed for S3 support")

    # Parse s3://bucket/key
    parsed = urlparse(source)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    logger.info(f"Downloading from S3: {bucket}/{key}")
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, str(dest_path))

    logger.info(f"Downloaded to: {dest_path}")
    return dest_path


def download_model(source: str, filename: str = "model.pt") -> Path:
    """
    Download model from any source.

    Always downloads fresh (no caching). For local files, validates
    existence and returns the path directly.

    Args:
        source: URL, local path, S3 path, or HuggingFace model ID
        filename: Filename for downloaded model (used for URL/S3/HF downloads)

    Returns:
        Path to the model file

    Raises:
        RuntimeError: If download fails
        FileNotFoundError: If local file doesn't exist

    Examples:
        # Local file
        path = download_model("/path/to/model.pt")

        # HTTP URL
        path = download_model("https://example.com/model.pt")

        # S3
        path = download_model("s3://my-bucket/models/yolo.pt")

        # HuggingFace
        path = download_model("ultralytics/yolov8n", filename="yolov8n.pt")
    """
    source_type = detect_source_type(source)

    # Local files: just validate and return
    if source_type == SourceType.LOCAL:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {source}")
        logger.info(f"Using local model: {path}")
        return path

    # Remote sources: download to temp directory
    download_dir = _get_download_dir()
    dest_path = download_dir / filename

    # Remove existing file if present (always fresh)
    if dest_path.exists():
        dest_path.unlink()

    try:
        if source_type == SourceType.URL:
            return _download_url(source, dest_path)
        elif source_type == SourceType.HUGGINGFACE:
            return _download_huggingface(source, dest_path)
        elif source_type == SourceType.S3:
            return _download_s3(source, dest_path)
        else:
            raise RuntimeError(f"Unsupported source type: {source_type}")

    except requests.exceptions.RequestException as e:
        if dest_path.exists():
            dest_path.unlink()  # Clean up partial download
        raise RuntimeError(f"Download failed: {e}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Model Downloader...")

    # Test source type detection
    assert detect_source_type("/path/to/model.pt") == SourceType.LOCAL
    assert detect_source_type("https://example.com/model.pt") == SourceType.URL
    assert detect_source_type("s3://bucket/model.pt") == SourceType.S3
    assert detect_source_type("ultralytics/yolov8n") == SourceType.HUGGINGFACE
    assert detect_source_type("https://huggingface.co/model") == SourceType.HUGGINGFACE
    print("  Source type detection works")

    # Test local file handling
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(b"fake model data")
        test_file = f.name

    try:
        path = download_model(test_file)
        assert path == Path(test_file)
        print("  Local file handling works")
    finally:
        os.unlink(test_file)

    # Test missing file error
    try:
        download_model("/nonexistent/model.pt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  Missing file error works")

    print("\nAll tests passed!")
