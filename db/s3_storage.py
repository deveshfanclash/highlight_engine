"""
S3 Storage

Simple storage for large inference outputs (segmentation masks, keypoints, embeddings).
Uses S3 for production, local files for testing.

Design Principles:
- Simple key structure: {match_id}/{service_id}/{frame_number}.{ext}
- Supports numpy arrays (masks), JSON (keypoints), binary (embeddings)
- Batched uploads to reduce API calls
- Local file fallback for testing without AWS

Usage:
    # Production (S3)
    storage = S3Storage(bucket="inference-outputs", match_id="match_123")
    storage.put_mask(frame_number=100, mask=mask_array, service_id="segmentation")

    # Testing (Local)
    storage = S3Storage(local_dir="/tmp/outputs", match_id="match_123")
    storage.put_mask(frame_number=100, mask=mask_array, service_id="segmentation")
"""

import os
import io
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from queue import Queue, Empty
from threading import Thread, Event

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class S3StorageConfig:
    """Configuration for S3 storage"""
    # S3 settings
    bucket: Optional[str] = None
    prefix: str = "inference"  # s3://bucket/inference/match_id/...
    region: str = "us-east-1"

    # AWS credentials (if not using IAM role)
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None

    # Local fallback (for testing)
    local_dir: Optional[str] = None

    # Batching (for many small files)
    batch_size: int = 10
    flush_interval_seconds: float = 5.0

    @classmethod
    def from_env(cls, bucket: Optional[str] = None) -> "S3StorageConfig":
        """Create config from environment variables"""
        return cls(
            bucket=bucket or os.getenv("S3_BUCKET"),
            prefix=os.getenv("S3_PREFIX", "inference"),
            region=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            local_dir=os.getenv("LOCAL_OUTPUT_DIR"),
        )


class S3Storage:
    """
    Simple storage for large inference outputs.

    Automatically uses local files if local_dir is set, otherwise S3.

    Key Structure:
        {prefix}/{match_id}/{service_id}/frame_{frame_number:08d}.{ext}

    Example:
        inference/match_123/segmentation/frame_00001500.npz
        inference/match_123/keypoints/frame_00001500.json
    """

    def __init__(
        self,
        match_id: str,
        bucket: Optional[str] = None,
        local_dir: Optional[str] = None,
        prefix: str = "inference",
        config: Optional[S3StorageConfig] = None
    ):
        """
        Initialize S3 storage.

        Args:
            match_id: Match identifier
            bucket: S3 bucket name (None = use local_dir)
            local_dir: Local directory for testing (takes precedence if set)
            prefix: S3 key prefix
            config: Full config object (overrides other args)
        """
        if config:
            self.config = config
        else:
            self.config = S3StorageConfig(
                bucket=bucket,
                local_dir=local_dir,
                prefix=prefix
            )

        self.match_id = match_id
        self._s3_client = None

        # Queue for background uploads
        self._queue: Optional[Queue] = None
        self._upload_thread: Optional[Thread] = None
        self._stop_event = Event()

        # Stats
        self._uploads_completed = 0
        self._bytes_uploaded = 0

    @property
    def use_local(self) -> bool:
        """Check if using local storage instead of S3"""
        return self.config.local_dir is not None

    @property
    def s3_client(self):
        """Lazy-load S3 client"""
        if self._s3_client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 is required for S3 support")

            kwargs = {"region_name": self.config.region}
            if self.config.aws_access_key and self.config.aws_secret_key:
                kwargs["aws_access_key_id"] = self.config.aws_access_key
                kwargs["aws_secret_access_key"] = self.config.aws_secret_key

            self._s3_client = boto3.client("s3", **kwargs)

        return self._s3_client

    def _build_key(self, service_id: str, frame_number: int, ext: str) -> str:
        """Build S3 key or local path"""
        filename = f"frame_{frame_number:08d}.{ext}"
        return f"{self.config.prefix}/{self.match_id}/{service_id}/{filename}"

    def _build_local_path(self, service_id: str, frame_number: int, ext: str) -> str:
        """Build local file path"""
        rel_path = f"{self.match_id}/{service_id}/frame_{frame_number:08d}.{ext}"
        full_path = os.path.join(self.config.local_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path

    # =========================================================================
    # MASK STORAGE (Numpy arrays - segmentation, depth, etc.)
    # =========================================================================

    def put_mask(
        self,
        frame_number: int,
        mask: np.ndarray,
        service_id: str = "segmentation",
        compress: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a segmentation mask or similar numpy array.

        Args:
            frame_number: Frame number
            mask: Numpy array (e.g., H x W or H x W x C)
            service_id: Service identifier
            compress: Use numpy compression (smaller files)
            metadata: Optional metadata dict

        Returns:
            Storage key/path
        """
        ext = "npz" if compress else "npy"
        buffer = io.BytesIO()

        if compress:
            np.savez_compressed(buffer, mask=mask, **(metadata or {}))
        else:
            np.save(buffer, mask)

        buffer.seek(0)
        data = buffer.getvalue()

        if self.use_local:
            path = self._build_local_path(service_id, frame_number, ext)
            with open(path, "wb") as f:
                f.write(data)
            self._uploads_completed += 1
            self._bytes_uploaded += len(data)
            return path
        else:
            key = self._build_key(service_id, frame_number, ext)
            self.s3_client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=data,
                ContentType="application/octet-stream"
            )
            self._uploads_completed += 1
            self._bytes_uploaded += len(data)
            return f"s3://{self.config.bucket}/{key}"

    def get_mask(
        self,
        frame_number: int,
        service_id: str = "segmentation",
        compress: bool = True
    ) -> Optional[np.ndarray]:
        """
        Retrieve a stored mask.

        Args:
            frame_number: Frame number
            service_id: Service identifier
            compress: Whether file is compressed

        Returns:
            Numpy array or None if not found
        """
        ext = "npz" if compress else "npy"

        try:
            if self.use_local:
                path = self._build_local_path(service_id, frame_number, ext)
                if not os.path.exists(path):
                    return None
                if compress:
                    data = np.load(path)
                    return data["mask"]
                else:
                    return np.load(path)
            else:
                key = self._build_key(service_id, frame_number, ext)
                response = self.s3_client.get_object(
                    Bucket=self.config.bucket,
                    Key=key
                )
                buffer = io.BytesIO(response["Body"].read())
                if compress:
                    data = np.load(buffer)
                    return data["mask"]
                else:
                    return np.load(buffer)

        except Exception as e:
            logger.error(f"Failed to get mask: {e}")
            return None

    # =========================================================================
    # JSON STORAGE (Keypoints, embeddings metadata, etc.)
    # =========================================================================

    def put_json(
        self,
        frame_number: int,
        data: Dict[str, Any],
        service_id: str = "keypoints"
    ) -> str:
        """
        Store JSON data (keypoints, small structured data).

        Args:
            frame_number: Frame number
            data: JSON-serializable dict
            service_id: Service identifier

        Returns:
            Storage key/path
        """
        json_bytes = json.dumps(data, default=self._json_encoder).encode("utf-8")

        if self.use_local:
            path = self._build_local_path(service_id, frame_number, "json")
            with open(path, "wb") as f:
                f.write(json_bytes)
            self._uploads_completed += 1
            self._bytes_uploaded += len(json_bytes)
            return path
        else:
            key = self._build_key(service_id, frame_number, "json")
            self.s3_client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=json_bytes,
                ContentType="application/json"
            )
            self._uploads_completed += 1
            self._bytes_uploaded += len(json_bytes)
            return f"s3://{self.config.bucket}/{key}"

    def get_json(
        self,
        frame_number: int,
        service_id: str = "keypoints"
    ) -> Optional[Dict[str, Any]]:
        """Retrieve stored JSON data."""
        try:
            if self.use_local:
                path = self._build_local_path(service_id, frame_number, "json")
                if not os.path.exists(path):
                    return None
                with open(path, "r") as f:
                    return json.load(f)
            else:
                key = self._build_key(service_id, frame_number, "json")
                response = self.s3_client.get_object(
                    Bucket=self.config.bucket,
                    Key=key
                )
                return json.loads(response["Body"].read().decode("utf-8"))

        except Exception as e:
            logger.error(f"Failed to get JSON: {e}")
            return None

    # =========================================================================
    # BINARY STORAGE (Embeddings, raw bytes, etc.)
    # =========================================================================

    def put_bytes(
        self,
        frame_number: int,
        data: bytes,
        service_id: str = "embeddings",
        ext: str = "bin"
    ) -> str:
        """
        Store raw bytes (embeddings, serialized tensors, etc.).

        Args:
            frame_number: Frame number
            data: Raw bytes
            service_id: Service identifier
            ext: File extension

        Returns:
            Storage key/path
        """
        if self.use_local:
            path = self._build_local_path(service_id, frame_number, ext)
            with open(path, "wb") as f:
                f.write(data)
            self._uploads_completed += 1
            self._bytes_uploaded += len(data)
            return path
        else:
            key = self._build_key(service_id, frame_number, ext)
            self.s3_client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=data,
                ContentType="application/octet-stream"
            )
            self._uploads_completed += 1
            self._bytes_uploaded += len(data)
            return f"s3://{self.config.bucket}/{key}"

    # =========================================================================
    # BACKGROUND UPLOAD (Queue-based)
    # =========================================================================

    def start_background_uploader(self):
        """Start background upload thread for async puts"""
        if self._upload_thread is not None:
            raise RuntimeError("Background uploader already running")

        self._queue = Queue()
        self._stop_event.clear()
        self._upload_thread = Thread(
            target=self._background_upload_loop,
            daemon=True
        )
        self._upload_thread.start()
        logger.info(f"Started S3Storage background uploader for {self.match_id}")

    def _background_upload_loop(self):
        """Background thread for processing upload queue"""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=self.config.flush_interval_seconds)

                if item is None:
                    break

                upload_type, kwargs = item
                if upload_type == "mask":
                    self.put_mask(**kwargs)
                elif upload_type == "json":
                    self.put_json(**kwargs)
                elif upload_type == "bytes":
                    self.put_bytes(**kwargs)

            except Empty:
                continue

        logger.info(f"S3Storage uploader stopped. Completed: {self._uploads_completed}")

    def queue_mask(self, **kwargs):
        """Queue mask for background upload"""
        if self._queue is None:
            raise RuntimeError("Background uploader not started")
        self._queue.put(("mask", kwargs))

    def queue_json(self, **kwargs):
        """Queue JSON for background upload"""
        if self._queue is None:
            raise RuntimeError("Background uploader not started")
        self._queue.put(("json", kwargs))

    def queue_bytes(self, **kwargs):
        """Queue bytes for background upload"""
        if self._queue is None:
            raise RuntimeError("Background uploader not started")
        self._queue.put(("bytes", kwargs))

    def stop(self, timeout: float = 10.0):
        """Stop background uploader and flush queue"""
        if self._upload_thread is None:
            return

        self._stop_event.set()
        if self._queue:
            self._queue.put(None)

        self._upload_thread.join(timeout=timeout)
        self._upload_thread = None
        self._queue = None

        logger.info(
            f"S3Storage stopped. Total: {self._uploads_completed} uploads, "
            f"{self._bytes_uploaded / 1024 / 1024:.2f} MB"
        )

    @staticmethod
    def _json_encoder(obj):
        """JSON encoder for numpy types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # =========================================================================
    # STATS
    # =========================================================================

    @property
    def stats(self) -> Dict[str, Any]:
        """Get upload statistics"""
        return {
            "uploads_completed": self._uploads_completed,
            "bytes_uploaded": self._bytes_uploaded,
            "mb_uploaded": round(self._bytes_uploaded / 1024 / 1024, 2),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_storage(
    match_id: str,
    bucket: Optional[str] = None,
    local_dir: Optional[str] = None,
    use_background: bool = False
) -> S3Storage:
    """
    Create storage instance based on configuration.

    Args:
        match_id: Match identifier
        bucket: S3 bucket (None = use local_dir or env)
        local_dir: Local directory for testing
        use_background: Start background uploader

    Returns:
        Configured S3Storage instance
    """
    # Local dir takes precedence
    if local_dir:
        storage = S3Storage(match_id=match_id, local_dir=local_dir)
    elif bucket:
        storage = S3Storage(match_id=match_id, bucket=bucket)
    else:
        # Try from environment
        config = S3StorageConfig.from_env()
        storage = S3Storage(match_id=match_id, config=config)

    if use_background:
        storage.start_background_uploader()

    return storage
