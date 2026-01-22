"""
Model Downloader

Downloads and caches model weights from various sources:
- Local files
- HTTP/HTTPS URLs
- HuggingFace Hub
- S3 (optional)

Usage:
    downloader = ModelDownloader(cache_dir="~/.cache/infer_models")

    # Download from URL
    path = downloader.download("https://example.com/model.pt", model_id="yolo_v8")

    # Check if cached
    if downloader.is_cached("yolo_v8"):
        path = downloader.get_cache_path("yolo_v8")

    # Clear cache
    downloader.clear_cache("yolo_v8")
"""

import os
import hashlib
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict
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


@dataclass
class DownloadResult:
    """Result of a download operation."""
    success: bool
    path: Optional[Path]
    source_type: SourceType
    cached: bool
    error: Optional[str] = None


@dataclass
class ModelDownloader:
    """
    Downloads and caches model weights from various sources.

    Attributes:
        cache_dir: Directory to store cached models
        timeout: Request timeout in seconds
        chunk_size: Download chunk size in bytes
    """
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "infer_models")
    timeout: int = 300
    chunk_size: int = 8192

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._checksums: Dict[str, str] = {}

    def detect_source_type(self, source: str) -> SourceType:
        """Detect the type of model source."""
        if os.path.exists(source):
            return SourceType.LOCAL

        if source.startswith("s3://"):
            return SourceType.S3

        if source.startswith(("http://", "https://")):
            # Check if it's HuggingFace
            parsed = urlparse(source)
            if "huggingface.co" in parsed.netloc:
                return SourceType.HUGGINGFACE
            return SourceType.URL

        # Check if it looks like a HuggingFace model ID (org/model)
        if "/" in source and not source.startswith("/"):
            parts = source.split("/")
            if len(parts) == 2 and all(p.replace("-", "").replace("_", "").isalnum() for p in parts):
                return SourceType.HUGGINGFACE

        # Assume local path that doesn't exist yet
        return SourceType.LOCAL

    def download(
        self,
        source: str,
        model_id: str,
        filename: str = "model.pt",
        force: bool = False,
        expected_checksum: Optional[str] = None,
    ) -> DownloadResult:
        """
        Download model from source.

        Args:
            source: URL, local path, or HuggingFace model ID
            model_id: Unique identifier for caching
            filename: Name for the cached file
            force: Force re-download even if cached
            expected_checksum: SHA256 checksum to validate (optional)

        Returns:
            DownloadResult with path and status
        """
        source_type = self.detect_source_type(source)
        cache_path = self.cache_dir / model_id / filename

        # Check cache first
        if not force and cache_path.exists():
            logger.info(f"Model cached: {cache_path}")
            if expected_checksum and not self.validate_checksum(cache_path, expected_checksum):
                logger.warning("Checksum mismatch, re-downloading")
            else:
                return DownloadResult(
                    success=True,
                    path=cache_path,
                    source_type=source_type,
                    cached=True
                )

        # Download based on source type
        try:
            if source_type == SourceType.LOCAL:
                return self._handle_local(source, cache_path)
            elif source_type == SourceType.URL:
                return self._download_url(source, cache_path, source_type)
            elif source_type == SourceType.HUGGINGFACE:
                return self._download_huggingface(source, cache_path)
            elif source_type == SourceType.S3:
                return self._download_s3(source, cache_path)
            else:
                return DownloadResult(
                    success=False,
                    path=None,
                    source_type=source_type,
                    cached=False,
                    error=f"Unsupported source type: {source_type}"
                )

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return DownloadResult(
                success=False,
                path=None,
                source_type=source_type,
                cached=False,
                error=str(e)
            )

    def _handle_local(self, source: str, cache_path: Path) -> DownloadResult:
        """Handle local file source."""
        source_path = Path(source)

        if not source_path.exists():
            return DownloadResult(
                success=False,
                path=None,
                source_type=SourceType.LOCAL,
                cached=False,
                error=f"Local file not found: {source}"
            )

        # Return source path directly (no copy needed)
        logger.info(f"Using local model: {source_path}")
        return DownloadResult(
            success=True,
            path=source_path,
            source_type=SourceType.LOCAL,
            cached=False
        )

    def _download_url(self, url: str, cache_path: Path, source_type: SourceType) -> DownloadResult:
        """Download from HTTP/HTTPS URL."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading: {url}")
        try:
            with requests.get(url, stream=True, timeout=self.timeout) as response:
                response.raise_for_status()

                # Get total size for progress
                total_size = int(response.headers.get('content-length', 0))

                with open(cache_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress every 10MB
                            if total_size > 0 and downloaded % (10 * 1024 * 1024) < self.chunk_size:
                                pct = (downloaded / total_size) * 100
                                logger.info(f"Download progress: {pct:.1f}%")

            logger.info(f"Downloaded to: {cache_path}")
            return DownloadResult(
                success=True,
                path=cache_path,
                source_type=source_type,
                cached=False
            )

        except requests.exceptions.RequestException as e:
            if cache_path.exists():
                cache_path.unlink()  # Clean up partial download
            raise RuntimeError(f"Download failed: {e}")

    def _download_huggingface(self, source: str, cache_path: Path) -> DownloadResult:
        """Download from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            # Fall back to URL if huggingface_hub not installed
            logger.warning("huggingface_hub not installed, trying URL fallback")

            # Check if source is already a URL
            if source.startswith("https://"):
                return self._download_url(source, cache_path, SourceType.HUGGINGFACE)

            # Convert model ID to URL
            # Format: org/model -> https://huggingface.co/org/model/resolve/main/model.pt
            parts = source.split("/")
            if len(parts) >= 2:
                url = f"https://huggingface.co/{source}/resolve/main/{cache_path.name}"
                return self._download_url(url, cache_path, SourceType.HUGGINGFACE)

            return DownloadResult(
                success=False,
                path=None,
                source_type=SourceType.HUGGINGFACE,
                cached=False,
                error="Cannot parse HuggingFace model ID"
            )

        # Use huggingface_hub library
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse source: could be "org/model" or "org/model/filename.pt"
        parts = source.split("/")
        if len(parts) == 2:
            repo_id = source
            filename = cache_path.name
        elif len(parts) >= 3:
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
        else:
            return DownloadResult(
                success=False,
                path=None,
                source_type=SourceType.HUGGINGFACE,
                cached=False,
                error=f"Invalid HuggingFace source: {source}"
            )

        logger.info(f"Downloading from HuggingFace: {repo_id}/{filename}")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=cache_path.parent,
            local_dir_use_symlinks=False
        )

        # Move/rename to expected cache path
        downloaded = Path(downloaded_path)
        if downloaded != cache_path:
            shutil.move(str(downloaded), str(cache_path))

        return DownloadResult(
            success=True,
            path=cache_path,
            source_type=SourceType.HUGGINGFACE,
            cached=False
        )

    def _download_s3(self, source: str, cache_path: Path) -> DownloadResult:
        """Download from S3."""
        try:
            import boto3
        except ImportError:
            return DownloadResult(
                success=False,
                path=None,
                source_type=SourceType.S3,
                cached=False,
                error="boto3 not installed for S3 support"
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse s3://bucket/key
        parsed = urlparse(source)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        logger.info(f"Downloading from S3: {bucket}/{key}")
        s3 = boto3.client('s3')
        s3.download_file(bucket, key, str(cache_path))

        return DownloadResult(
            success=True,
            path=cache_path,
            source_type=SourceType.S3,
            cached=False
        )

    def is_cached(self, model_id: str, filename: str = "model.pt") -> bool:
        """Check if model is cached."""
        cache_path = self.cache_dir / model_id / filename
        return cache_path.exists()

    def get_cache_path(self, model_id: str, filename: str = "model.pt") -> Path:
        """Get path to cached model."""
        return self.cache_dir / model_id / filename

    def clear_cache(self, model_id: Optional[str] = None):
        """
        Clear cached models.

        Args:
            model_id: Specific model to clear, or None to clear all
        """
        if model_id:
            cache_path = self.cache_dir / model_id
            if cache_path.exists():
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache: {model_id}")
        else:
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
            logger.info("Cleared all model cache")

    def validate_checksum(self, file_path: Path, expected: str) -> bool:
        """
        Validate file SHA256 checksum.

        Args:
            file_path: Path to file
            expected: Expected SHA256 hash (hex string)

        Returns:
            True if checksum matches
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        actual = sha256.hexdigest()
        return actual == expected.lower()

    def get_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def list_cached(self) -> Dict[str, Path]:
        """List all cached models."""
        cached = {}
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                for f in item.iterdir():
                    if f.is_file():
                        cached[item.name] = f
                        break
        return cached


# Global downloader instance
_default_downloader: Optional[ModelDownloader] = None


def get_downloader(cache_dir: Optional[Path] = None) -> ModelDownloader:
    """Get or create the default model downloader."""
    global _default_downloader

    if _default_downloader is None or cache_dir is not None:
        _default_downloader = ModelDownloader(
            cache_dir=cache_dir or Path.home() / ".cache" / "infer_models"
        )

    return _default_downloader


def download_model(
    source: str,
    model_id: str,
    filename: str = "model.pt",
    force: bool = False,
) -> Path:
    """
    Convenience function to download a model.

    Args:
        source: URL, local path, or HuggingFace model ID
        model_id: Unique identifier for caching
        filename: Name for the cached file
        force: Force re-download even if cached

    Returns:
        Path to the model file

    Raises:
        RuntimeError: If download fails
    """
    downloader = get_downloader()
    result = downloader.download(source, model_id, filename, force)

    if not result.success:
        raise RuntimeError(f"Failed to download model: {result.error}")

    return result.path


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import tempfile

    print("Testing ModelDownloader...")

    # Create downloader with temp cache
    with tempfile.TemporaryDirectory() as tmp:
        downloader = ModelDownloader(cache_dir=Path(tmp))

        # Test source type detection
        assert downloader.detect_source_type("/path/to/model.pt") == SourceType.LOCAL
        assert downloader.detect_source_type("https://example.com/model.pt") == SourceType.URL
        assert downloader.detect_source_type("s3://bucket/model.pt") == SourceType.S3
        assert downloader.detect_source_type("ultralytics/yolov8n") == SourceType.HUGGINGFACE
        assert downloader.detect_source_type("https://huggingface.co/model") == SourceType.HUGGINGFACE
        print("  Source type detection works")

        # Test local file handling
        test_file = Path(tmp) / "test_model.pt"
        test_file.write_text("fake model data")

        result = downloader.download(str(test_file), "test_local")
        assert result.success
        assert result.source_type == SourceType.LOCAL
        print("  Local file handling works")

        # Test caching
        assert not downloader.is_cached("nonexistent")
        print("  Cache check works")

        # Test clear cache
        downloader.clear_cache("test_local")
        print("  Cache clear works")

        # Test checksum
        checksum = downloader.get_checksum(test_file)
        assert len(checksum) == 64  # SHA256 hex
        assert downloader.validate_checksum(test_file, checksum)
        print("  Checksum validation works")

    print("\nAll tests passed!")
