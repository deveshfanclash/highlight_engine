"""
DynamoDB Writer

Batched writer for inference results with background queue processing.

Key patterns:
- Abstract BaseWriter with shared queue logic
- Background thread for async writes
- Batch buffering with configurable size and flush interval
- Basic metrics for observability
"""

import os
import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from queue import Queue, Empty
from threading import Thread, Event, Lock

logger = logging.getLogger(__name__)


# =============================================================================
# METRICS
# =============================================================================

class WriterMetrics:
    """Basic metrics for writer observability."""

    def __init__(self):
        self.items_queued: int = 0
        self.items_written: int = 0
        self.batches_written: int = 0
        self.write_errors: int = 0
        self.total_write_time_ms: float = 0.0
        self._lock = Lock()

    def record_write(self, items_count: int, elapsed_ms: float):
        """Record a successful batch write."""
        with self._lock:
            self.items_written += items_count
            self.batches_written += 1
            self.total_write_time_ms += elapsed_ms

    def record_error(self):
        """Record a write error."""
        with self._lock:
            self.write_errors += 1

    def record_queued(self, count: int = 1):
        """Record items added to queue."""
        with self._lock:
            self.items_queued += count

    @property
    def avg_write_latency_ms(self) -> float:
        """Average write latency per batch."""
        # No lock needed - reading is atomic for simple types
        if self.batches_written == 0:
            return 0.0
        return self.total_write_time_ms / self.batches_written

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        with self._lock:
            avg_latency = 0.0
            if self.batches_written > 0:
                avg_latency = self.total_write_time_ms / self.batches_written
            return {
                "items_queued": self.items_queued,
                "items_written": self.items_written,
                "batches_written": self.batches_written,
                "write_errors": self.write_errors,
                "avg_write_latency_ms": round(avg_latency, 2),
            }


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DynamoDBWriterConfig:
    """Configuration for DynamoDB writer."""
    table_name: str
    region: str = "us-east-1"

    # AWS credentials (if not using IAM role)
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None

    # Batching configuration
    batch_size: int = 12
    flush_interval_ms: int = 250

    @classmethod
    def from_infra_config(cls, table_name: str) -> "DynamoDBWriterConfig":
        """Create config from InfraConfig."""
        from config.environment import get_infra_config
        infra = get_infra_config()

        return cls(
            table_name=table_name,
            region=infra.aws_region,
            aws_access_key=infra.aws_access_key,
            aws_secret_key=infra.aws_secret_key,
        )


# =============================================================================
# ABSTRACT BASE WRITER
# =============================================================================

class BaseWriter(ABC):
    """
    Abstract base class for writers with shared queue/batching logic.

    Subclasses only need to implement:
    - _write_batch(): Write a batch of items to storage
    - _write_immediate(): Write a single item immediately
    """

    def __init__(self, batch_size: int = 12, flush_interval_ms: int = 250):
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms

        # Internal state
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush = time.time()

        # Queue mode
        self._queue: Optional[Queue] = None
        self._writer_thread: Optional[Thread] = None
        self._stop_event = Event()

        # Metrics
        self.metrics = WriterMetrics()

    # =========================================================================
    # ABSTRACT METHODS (subclasses implement these)
    # =========================================================================

    @abstractmethod
    def _write_batch(self, items: List[Dict[str, Any]]) -> int:
        """
        Write a batch of items to storage.

        Args:
            items: List of prepared items to write

        Returns:
            Number of items successfully written
        """
        pass

    @abstractmethod
    def _write_immediate(self, item: Dict[str, Any]) -> bool:
        """
        Write a single item immediately (bypassing queue/buffer).

        Args:
            item: Item to write

        Returns:
            True if successful
        """
        pass

    # =========================================================================
    # SHARED LOGIC
    # =========================================================================

    def _should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        if len(self._buffer) >= self.batch_size:
            return True

        elapsed_ms = (time.time() - self._last_flush) * 1000
        if elapsed_ms >= self.flush_interval_ms and self._buffer:
            return True

        return False

    def _flush(self) -> int:
        """Flush buffered items to storage."""
        if not self._buffer:
            return 0

        items_to_write = self._buffer.copy()
        self._buffer.clear()

        start = time.perf_counter()
        written = self._write_batch(items_to_write)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.metrics.record_write(written, elapsed_ms)
        self._last_flush = time.time()

        return written

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def start(self):
        """Start background writer thread."""
        if self._writer_thread is not None:
            raise RuntimeError("Writer already running")

        self._queue = Queue()
        self._stop_event.clear()
        self._writer_thread = Thread(
            target=self._background_loop,
            daemon=True
        )
        self._writer_thread.start()
        logger.info(f"Started background writer: {self.__class__.__name__}")

    def _background_loop(self):
        """Background thread loop for processing queue."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=5.0)

                if item is None:
                    # Termination signal
                    break

                self._buffer.append(item)

                if self._should_flush():
                    self._flush()

            except Empty:
                # Timeout - check if we should flush
                if self._should_flush():
                    self._flush()
                continue

        # Final flush
        self._flush()
        logger.info(f"Background writer stopped: {self.__class__.__name__}")

    def queue_item(self, item: Dict[str, Any]):
        """
        Add item to write queue for background processing.

        Args:
            item: Item to write
        """
        if self._queue is None:
            raise RuntimeError("Writer not started. Call start() first.")

        self._queue.put(item)
        self.metrics.record_queued()

    def write_item(self, item: Dict[str, Any], immediate: bool = False) -> bool:
        """
        Write a single item.

        Args:
            item: Item to write
            immediate: Must be True (queued writes use queue_item)

        Returns:
            True if successful
        """
        if not immediate:
            raise ValueError(
                "write_item() requires immediate=True. "
                "Use queue_item() for queued writes."
            )
        return self._write_immediate(item)

    def stop(self, timeout: float = 10.0):
        """Stop background writer and flush remaining items."""
        if self._writer_thread is None:
            self._flush()
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
        self._flush()

        # Log metrics
        logger.info(f"Writer metrics: {self.metrics.to_dict()}")


# =============================================================================
# DYNAMODB WRITER
# =============================================================================

class DynamoDBWriter(BaseWriter):
    """
    DynamoDB writer with batched background writes.

    Usage:
        config = DynamoDBWriterConfig.from_infra_config("my_table")
        writer = DynamoDBWriter(config)
        writer.start()

        writer.queue_item({"pk": "...", "sk": 1, ...})
        writer.stop()
    """

    def __init__(self, config: DynamoDBWriterConfig):
        super().__init__(
            batch_size=config.batch_size,
            flush_interval_ms=config.flush_interval_ms,
        )
        self.config = config
        self._table = None

    def _get_table(self):
        """Lazy-load DynamoDB table resource."""
        if self._table is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 is required for DynamoDB support")

            kwargs = {"region_name": self.config.region}
            if self.config.aws_access_key and self.config.aws_secret_key:
                kwargs["aws_access_key_id"] = self.config.aws_access_key
                kwargs["aws_secret_access_key"] = self.config.aws_secret_key

            dynamodb = boto3.resource("dynamodb", **kwargs)
            self._table = dynamodb.Table(self.config.table_name)

        return self._table

    @staticmethod
    def _convert_to_decimal(obj: Any) -> Any:
        """Recursively convert floats to Decimals for DynamoDB."""
        from decimal import Decimal

        if isinstance(obj, float):
            return Decimal(str(round(obj, 6)))
        elif isinstance(obj, dict):
            return {k: DynamoDBWriter._convert_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DynamoDBWriter._convert_to_decimal(item) for item in obj]
        return obj

    def _prepare_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare item for DynamoDB (convert floats, add metadata)."""
        prepared = self._convert_to_decimal(item)

        if "written_at" not in prepared:
            prepared["written_at"] = datetime.utcnow().isoformat() + "Z"

        return prepared

    def _write_batch(self, items: List[Dict[str, Any]]) -> int:
        """Write batch to DynamoDB."""
        if not items:
            return 0

        table = self._get_table()
        written = 0

        try:
            with table.batch_writer() as writer:
                for item in items:
                    try:
                        prepared = self._prepare_item(item)
                        writer.put_item(Item=prepared)
                        written += 1
                    except Exception as e:
                        logger.error(f"Failed to prepare item: {e}")
                        self.metrics.record_error()

            logger.debug(f"Batch wrote {written} items to DynamoDB")

        except Exception as e:
            logger.error(f"Batch write failed: {e}")
            self.metrics.record_error()

            # Fallback to individual writes
            logger.info("Falling back to individual writes...")
            for item in items:
                try:
                    prepared = self._prepare_item(item)
                    table.put_item(Item=prepared)
                    written += 1
                except Exception as e2:
                    logger.error(f"Individual write failed: {e2}")
                    self.metrics.record_error()

        return written

    def _write_immediate(self, item: Dict[str, Any]) -> bool:
        """Write single item immediately to DynamoDB."""
        try:
            table = self._get_table()
            prepared = self._prepare_item(item)
            table.put_item(Item=prepared)
            self.metrics.record_write(1, 0)
            return True
        except Exception as e:
            logger.error(f"Immediate write failed: {e}")
            self.metrics.record_error()
            return False


# =============================================================================
# LOCAL FILE WRITER
# =============================================================================

class LocalFileWriter(BaseWriter):
    """
    Local file writer for testing without DynamoDB.

    Writes to JSON Lines files (.jsonl).
    """

    def __init__(
        self,
        output_dir: str,
        match_id: str,
        service_id: str = "inference",
        batch_size: int = 12,
        flush_interval_ms: int = 250,
    ):
        super().__init__(
            batch_size=batch_size,
            flush_interval_ms=flush_interval_ms,
        )

        self.output_dir = output_dir
        self.match_id = match_id
        self.service_id = service_id

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

    def _write_batch(self, items: List[Dict[str, Any]]) -> int:
        """Write batch to file."""
        if not items:
            return 0

        try:
            with open(self._output_file, "a") as f:
                for item in items:
                    f.write(json.dumps(item, default=str) + "\n")

            logger.debug(f"Wrote {len(items)} items to {self._output_file}")
            return len(items)

        except Exception as e:
            logger.error(f"File write failed: {e}")
            self.metrics.record_error()
            return 0

    def _write_immediate(self, item: Dict[str, Any]) -> bool:
        """Write single item immediately to file."""
        try:
            with open(self._output_file, "a") as f:
                f.write(json.dumps(item, default=str) + "\n")
            self.metrics.record_write(1, 0)
            return True
        except Exception as e:
            logger.error(f"Immediate file write failed: {e}")
            self.metrics.record_error()
            return False

    def stop(self, timeout: float = 10.0):
        """Stop writer and log final stats."""
        super().stop(timeout)

        # Log file stats
        if os.path.exists(self._output_file):
            line_count = sum(1 for _ in open(self._output_file))
            logger.info(f"Total lines in file: {line_count}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_writer(
    table_name: str = "inference_results",
    local_mode: bool = False,
    local_output_dir: Optional[str] = None,
    match_id: str = "test",
    service_id: str = "inference",
    batch_size: int = 12,
    flush_interval_ms: int = 250,
) -> BaseWriter:
    """
    Factory function to create and start appropriate writer.

    Args:
        table_name: DynamoDB table name (for DynamoDB mode)
        local_mode: If True, use LocalFileWriter
        local_output_dir: Output directory for local mode
        match_id: Match ID (for local writer filename)
        service_id: Service ID (for local writer filename)
        batch_size: Items per batch before flush
        flush_interval_ms: Max time between flushes

    Returns:
        Started BaseWriter instance
    """
    if local_mode or local_output_dir:
        output_dir = local_output_dir
        if not output_dir:
            from config.environment import get_infra_config
            output_dir = str(get_infra_config(local_mode=True).output_path)

        writer = LocalFileWriter(
            output_dir=output_dir,
            match_id=match_id,
            service_id=service_id,
            batch_size=batch_size,
            flush_interval_ms=flush_interval_ms,
        )
    else:
        config = DynamoDBWriterConfig.from_infra_config(table_name)
        config.batch_size = batch_size
        config.flush_interval_ms = flush_interval_ms
        writer = DynamoDBWriter(config)

    # Always start the background writer
    writer.start()

    return writer


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil

    print("Testing BaseWriter implementation...\n")

    # Test LocalFileWriter
    print("1. Testing LocalFileWriter...")
    test_dir = tempfile.mkdtemp()

    try:
        writer = LocalFileWriter(
            output_dir=test_dir,
            match_id="test_match",
            service_id="od",
            batch_size=3,
            flush_interval_ms=100,
        )
        writer.start()

        # Queue items
        for i in range(10):
            writer.queue_item({
                "frame": i,
                "detections": [{"class": "ball", "conf": 0.95}],
                "float_value": 0.123456789,
            })

        # Wait a bit for processing
        time.sleep(0.5)

        # Test immediate write
        result = writer.write_item({"status": "completed"}, immediate=True)
        assert result, "Immediate write should succeed"

        # Test error on non-immediate
        try:
            writer.write_item({"bad": "call"}, immediate=False)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"   Correctly rejected non-immediate write_item: {e}")

        writer.stop()

        # Verify output
        output_file = os.path.join(test_dir, "test_match_od.jsonl")
        assert os.path.exists(output_file), "Output file should exist"

        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 11, f"Expected 11 lines, got {len(lines)}"
        print(f"   Wrote {len(lines)} items successfully")

        # Verify metrics
        metrics = writer.metrics.to_dict()
        print(f"   Metrics: {metrics}")
        assert metrics["items_queued"] == 10
        assert metrics["items_written"] == 11  # 10 queued + 1 immediate

        print("   LocalFileWriter: PASSED\n")

    finally:
        shutil.rmtree(test_dir)

    # Test metrics
    print("2. Testing WriterMetrics...")
    metrics = WriterMetrics()

    metrics.record_queued(5)
    metrics.record_write(5, 10.0)
    metrics.record_write(5, 20.0)
    metrics.record_error()

    d = metrics.to_dict()
    assert d["items_queued"] == 5
    assert d["items_written"] == 10
    assert d["batches_written"] == 2
    assert d["write_errors"] == 1
    assert d["avg_write_latency_ms"] == 15.0

    print(f"   Metrics: {d}")
    print("   WriterMetrics: PASSED\n")

    # Test batch flushing
    print("3. Testing batch flush behavior...")
    test_dir = tempfile.mkdtemp()

    try:
        writer = LocalFileWriter(
            output_dir=test_dir,
            match_id="batch_test",
            service_id="test",
            batch_size=5,
            flush_interval_ms=1000,  # Long interval
        )
        writer.start()

        # Queue exactly batch_size items - should trigger flush
        for i in range(5):
            writer.queue_item({"index": i})

        time.sleep(0.2)  # Let it process

        # Check that flush happened
        assert writer.metrics.batches_written >= 1, "Should have flushed at least once"
        print(f"   Batches written after 5 items: {writer.metrics.batches_written}")

        writer.stop()
        print("   Batch flush: PASSED\n")

    finally:
        shutil.rmtree(test_dir)

    print("=" * 50)
    print("All tests PASSED!")
