"""
DynamoDB Writer

Batched writer for inference results with configurable flush intervals.
Extracted and refactored from old_inference_code_for_reference/src/database/dynamodb_writer.py

Key patterns preserved:
- Batch buffering (WRITE_DELAY frames)
- Time-based flush (WRITE_INTERVAL)
- Decimal conversion for DynamoDB
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from decimal import Decimal
from datetime import datetime
from queue import Queue, Empty
from threading import Thread, Event

logger = logging.getLogger(__name__)


@dataclass
class DynamoDBWriterConfig:
    """Configuration for DynamoDB writer"""
    table_name: str
    region: str = "us-east-1"

    # AWS credentials (if not using IAM role)
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None

    # Batching configuration
    batch_size: int = 12  # Number of items to buffer before write
    flush_interval_ms: int = 250  # Max time between flushes

    # Queue configuration
    queue_timeout: float = 5.0  # Timeout for queue.get()

    @classmethod
    def from_env(cls, table_name: str) -> "DynamoDBWriterConfig":
        """Create config from environment variables"""
        return cls(
            table_name=table_name,
            region=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY_DB"),
            aws_secret_key=os.getenv("AWS_SECRET_KEY_DB"),
        )


class DynamoDBWriter:
    """
    Batched DynamoDB writer for inference results.

    Supports two modes:
    1. Direct write: Call write_item() directly
    2. Queue-based: Push to queue, writer flushes in background

    Usage (Direct):
        writer = DynamoDBWriter(config)
        writer.write_item({"pk": "...", "sk": 1, ...})
        writer.flush()  # Ensure all items are written

    Usage (Queue-based):
        writer = DynamoDBWriter(config)
        writer.start_background_writer()
        writer.queue_item({"pk": "...", "sk": 1, ...})
        writer.stop()  # Stop and flush remaining
    """

    def __init__(self, config: DynamoDBWriterConfig):
        self.config = config
        self._table = None
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush = time.time()

        # Queue mode
        self._queue: Optional[Queue] = None
        self._writer_thread: Optional[Thread] = None
        self._stop_event = Event()

    def _get_table(self):
        """Lazy-load DynamoDB table resource"""
        if self._table is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 is required for DynamoDB support")

            # Build connection kwargs
            kwargs = {"region_name": self.config.region}
            if self.config.aws_access_key and self.config.aws_secret_key:
                kwargs["aws_access_key_id"] = self.config.aws_access_key
                kwargs["aws_secret_access_key"] = self.config.aws_secret_key

            dynamodb = boto3.resource("dynamodb", **kwargs)
            self._table = dynamodb.Table(self.config.table_name)

        return self._table

    @staticmethod
    def _convert_to_decimal(obj: Any) -> Any:
        """
        Recursively convert floats to Decimals for DynamoDB.

        DynamoDB doesn't support float, so we convert to Decimal.
        """
        if isinstance(obj, float):
            return Decimal(str(round(obj, 6)))
        elif isinstance(obj, dict):
            return {k: DynamoDBWriter._convert_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DynamoDBWriter._convert_to_decimal(item) for item in obj]
        return obj

    def _prepare_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare item for DynamoDB (convert floats, add metadata)"""
        prepared = self._convert_to_decimal(item)

        # Add write timestamp if not present
        if "written_at" not in prepared:
            prepared["written_at"] = datetime.utcnow().isoformat() + "Z"

        return prepared

    def _should_flush(self) -> bool:
        """Check if buffer should be flushed"""
        if len(self._buffer) >= self.config.batch_size:
            return True

        elapsed_ms = (time.time() - self._last_flush) * 1000
        if elapsed_ms >= self.config.flush_interval_ms and self._buffer:
            return True

        return False

    def flush(self) -> int:
        """
        Flush buffered items to DynamoDB.

        Returns:
            Number of items written
        """
        if not self._buffer:
            return 0

        items_to_write = self._buffer.copy()
        self._buffer.clear()

        written = self.write_items_batch(items_to_write)
        self._last_flush = time.time()

        return written

    def write_items_batch(
        self,
        items: List[Dict[str, Any]],
        overwrite_keys: Optional[List[str]] = None
    ) -> int:
        """
        Write multiple items using DynamoDB batch_writer (efficient for bulk writes).

        Uses batch_writer() which:
        - Automatically batches items (max 25 per request, DynamoDB limit)
        - Handles retries for unprocessed items
        - Much more efficient than individual put_item() calls

        Args:
            items: List of items to write
            overwrite_keys: Primary key fields for upsert behavior
                           (e.g., ["match_id", "ptstime"] for HLS metadata)

        Returns:
            Number of items written
        """
        if not items:
            return 0

        table = self._get_table()
        written = 0

        try:
            # Use batch_writer for efficient bulk writes
            # overwrite_by_pkeys allows upsert behavior (replace if exists)
            batch_kwargs = {}
            if overwrite_keys:
                batch_kwargs["overwrite_by_pkeys"] = overwrite_keys

            with table.batch_writer(**batch_kwargs) as writer:
                for item in items:
                    try:
                        prepared = self._prepare_item(item)
                        writer.put_item(Item=prepared)
                        written += 1
                    except Exception as e:
                        logger.error(f"Failed to prepare item for batch: {e}")

            logger.debug(f"Batch wrote {written} items to DynamoDB")

        except Exception as e:
            logger.error(f"Batch write failed: {e}")
            # Fallback to individual writes if batch fails
            logger.info("Falling back to individual writes...")
            for item in items:
                try:
                    prepared = self._prepare_item(item)
                    table.put_item(Item=prepared)
                    written += 1
                except Exception as e2:
                    logger.error(f"Individual write also failed: {e2}")

        return written

    def write_item(self, item: Dict[str, Any], immediate: bool = False) -> bool:
        """
        Write a single item (buffered by default).

        Args:
            item: Item to write
            immediate: If True, write immediately without buffering

        Returns:
            True if item was queued/written successfully
        """
        if immediate:
            try:
                table = self._get_table()
                prepared = self._prepare_item(item)
                table.put_item(Item=prepared)
                return True
            except Exception as e:
                logger.error(f"Failed to write item: {e}")
                return False

        self._buffer.append(item)

        if self._should_flush():
            self.flush()

        return True

    # =========================================================================
    # QUEUE-BASED WRITING (Background Thread)
    # =========================================================================

    def start_background_writer(self):
        """Start background writer thread for queue-based writing"""
        if self._writer_thread is not None:
            raise RuntimeError("Background writer already running")

        self._queue = Queue()
        self._stop_event.clear()
        self._writer_thread = Thread(
            target=self._background_writer_loop,
            daemon=True
        )
        self._writer_thread.start()
        logger.info("Started background DynamoDB writer")

    def _background_writer_loop(self):
        """Background thread loop for processing queue"""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=self.config.queue_timeout)

                if item is None:
                    # Termination signal
                    logger.info("Received termination signal")
                    break

                self.write_item(item)

            except Empty:
                # Timeout - check if we should flush
                if self._should_flush():
                    self.flush()
                continue

        # Final flush
        self.flush()
        logger.info("Background writer stopped")

    def queue_item(self, item: Dict[str, Any]):
        """
        Add item to write queue (for background writing).

        Args:
            item: Item to write
        """
        if self._queue is None:
            raise RuntimeError("Background writer not started. Call start_background_writer() first.")

        self._queue.put(item)

    def stop(self, timeout: float = 10.0):
        """
        Stop background writer and flush remaining items.

        Args:
            timeout: Max time to wait for writer to stop
        """
        if self._writer_thread is None:
            self.flush()
            return

        # Send termination signal
        self._stop_event.set()
        if self._queue:
            self._queue.put(None)

        # Wait for thread to finish
        self._writer_thread.join(timeout=timeout)

        if self._writer_thread.is_alive():
            logger.warning("Background writer did not stop cleanly")

        self._writer_thread = None
        self._queue = None

        # Final flush
        self.flush()

    # =========================================================================
    # CONVENIENCE METHODS FOR INFERENCE OUTPUT
    # =========================================================================

    def write_inference_result(
        self,
        match_id: str,
        service_id: str,
        model_id: str,
        frame_number: int,
        timestamp_ms: int,
        detections: List[Dict[str, Any]],
        processing_time_ms: int = 0
    ):
        """
        Write inference result in standard format.

        Args:
            match_id: Match identifier
            service_id: Service identifier
            model_id: Model that produced detections
            frame_number: Frame number
            timestamp_ms: Frame timestamp in milliseconds
            detections: List of detection dicts with class_name, class_id, confidence, bbox
            processing_time_ms: Processing time for this frame
        """
        item = {
            "pk": f"{match_id}#{service_id}",
            "sk": frame_number,
            "match_id": match_id,
            "service_id": service_id,
            "model_id": model_id,
            "frame_number": frame_number,
            "timestamp_ms": timestamp_ms,
            "detections": detections,
            "processing_time_ms": processing_time_ms,
        }

        self.write_item(item)

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_inference_writer(
    table_name: str,
    use_background: bool = True
) -> DynamoDBWriter:
    """
    Create a DynamoDB writer for inference results.

    Args:
        table_name: DynamoDB table name
        use_background: If True, start background writer thread

    Returns:
        Configured DynamoDBWriter
    """
    config = DynamoDBWriterConfig.from_env(table_name)
    writer = DynamoDBWriter(config)

    if use_background:
        writer.start_background_writer()

    return writer


# =============================================================================
# LOCAL FILE WRITER (For Testing)
# =============================================================================

class LocalFileWriter:
    """
    Local file writer for testing without DynamoDB.

    Writes output to JSON Lines files (.jsonl) in a specified directory.
    Drop-in replacement for DynamoDBWriter in test mode.

    Usage:
        writer = LocalFileWriter("/tmp/test_output", "match_123")
        writer.start_background_writer()
        writer.queue_item({"frame_number": 1, "detections": [...]})
        writer.stop()

        # Output written to: /tmp/test_output/match_123_inference.jsonl
    """

    def __init__(self, output_dir: str, match_id: str, service_id: str = "inference"):
        self.output_dir = output_dir
        self.match_id = match_id
        self.service_id = service_id

        self._buffer: List[Dict[str, Any]] = []
        self._queue: Optional[Queue] = None
        self._writer_thread: Optional[Thread] = None
        self._stop_event = Event()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        self._output_file = os.path.join(
            output_dir,
            f"{match_id}_{service_id}.jsonl"
        )

        # Clear existing file
        if os.path.exists(self._output_file):
            os.remove(self._output_file)

        logger.info(f"LocalFileWriter initialized: {self._output_file}")

    def write_item(self, item: Dict[str, Any], immediate: bool = False) -> bool:
        """Write item to buffer or immediately to file"""
        import json

        if immediate:
            with open(self._output_file, "a") as f:
                f.write(json.dumps(item, default=str) + "\n")
            return True

        self._buffer.append(item)

        if len(self._buffer) >= 12:  # Same batch size as DynamoDB
            self.flush()

        return True

    def flush(self) -> int:
        """Flush buffer to file"""
        if not self._buffer:
            return 0

        count = self.write_items_batch(self._buffer)
        self._buffer.clear()
        return count

    def write_items_batch(
        self,
        items: List[Dict[str, Any]],
        overwrite_keys: Optional[List[str]] = None
    ) -> int:
        """
        Write multiple items to file in a single operation.

        Args:
            items: List of items to write
            overwrite_keys: Ignored for local files (included for API compatibility)

        Returns:
            Number of items written
        """
        import json

        if not items:
            return 0

        with open(self._output_file, "a") as f:
            for item in items:
                f.write(json.dumps(item, default=str) + "\n")

        return len(items)

    def start_background_writer(self):
        """Start background writer thread"""
        if self._writer_thread is not None:
            raise RuntimeError("Background writer already running")

        self._queue = Queue()
        self._stop_event.clear()
        self._writer_thread = Thread(
            target=self._background_writer_loop,
            daemon=True
        )
        self._writer_thread.start()
        logger.info(f"Started LocalFileWriter background thread: {self._output_file}")

    def _background_writer_loop(self):
        """Background thread loop"""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=5.0)
                if item is None:
                    break
                self.write_item(item)
            except Empty:
                if self._buffer:
                    self.flush()
                continue

        self.flush()
        logger.info(f"LocalFileWriter stopped: {self._output_file}")

    def queue_item(self, item: Dict[str, Any]):
        """Add item to write queue"""
        if self._queue is None:
            raise RuntimeError("Background writer not started")
        self._queue.put(item)

    def stop(self, timeout: float = 10.0):
        """Stop writer and flush remaining"""
        if self._writer_thread is None:
            self.flush()
            return

        self._stop_event.set()
        if self._queue:
            self._queue.put(None)

        self._writer_thread.join(timeout=timeout)
        self._writer_thread = None
        self._queue = None
        self.flush()

        # Log final stats
        if os.path.exists(self._output_file):
            line_count = sum(1 for _ in open(self._output_file))
            logger.info(f"Total items written: {line_count} to {self._output_file}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def create_writer(
    table_name: str = "inference_results",
    use_background: bool = True,
    local_output_dir: Optional[str] = None,
    match_id: str = "test",
    service_id: str = "inference"
):
    """
    Factory function to create appropriate writer based on mode.

    Args:
        table_name: DynamoDB table name (for DynamoDB mode)
        use_background: Use background writer thread
        local_output_dir: If set, use LocalFileWriter instead of DynamoDB
        match_id: Match ID (for local writer)
        service_id: Service ID (for local writer)

    Returns:
        DynamoDBWriter or LocalFileWriter
    """
    if local_output_dir:
        writer = LocalFileWriter(local_output_dir, match_id, service_id)
    else:
        config = DynamoDBWriterConfig.from_env(table_name)
        writer = DynamoDBWriter(config)

    if use_background:
        writer.start_background_writer()

    return writer
