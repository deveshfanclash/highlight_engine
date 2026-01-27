"""
Base Service

Abstract base class for all inference services.
Defines the common lifecycle and interface that all services must implement.

ServiceConfig defines runtime configuration (stream, match ID, device, etc.)
Extended by specific configs (ODServiceConfig, PoseServiceConfig).
"""

import logging
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
from datetime import datetime

from config.schemas import InputType
from input_handlers import FrameInputHandler, FrameInputPacket
from db.dynamo import BaseWriter, create_writer
from core.batch_accumulator import BatchAccumulator, Batch

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """
    Runtime configuration for running a service.

    Extended by specific configs (ODServiceConfig, PoseServiceConfig).
    """
    # Identifiers
    match_id: str
    service_id: str  # Unique ID for this service instance (e.g., "od_football_v2")

    # Input configuration
    input_source: str  # Stream URL or file path
    input_type: InputType = InputType.HLS

    # Processing settings
    target_width: Optional[int] = None  # None = use source resolution
    target_height: Optional[int] = None
    frame_skip: int = 1  # Process every Nth frame
    start_frame: int = 0  # For resume support
    start_segment: int = 1  # For HLS resume support

    # Batching settings (for GPU efficiency)
    # batch_size > 1 enables batched processing for ~3-8x throughput
    inference_batch_size: int = 1  # Number of frames to process at once

    # Multi-worker settings (for process-based parallelism)
    num_workers: int = 1  # Number of worker processes
    worker_queue_size: int = 0  # Queue size per worker (0 = auto: num_workers * 4)

    # Buffer settings (for async frame extraction)
    enable_buffering: Optional[bool] = None  # None = auto-detect based on source type
    buffer_size: int = 30  # Buffer size for frame buffering
    buffer_mode: str = "drop_old"  # "fifo" or "drop_old"

    # Device settings
    device: str = "cpu"  # "cpu", "cuda:0", "cuda:1", etc.

    # Database settings
    db_table_name: str = "inference_results"
    db_batch_size: int = 12
    db_flush_interval_ms: int = 250

    # Local mode settings
    local_mode: bool = False  # If True, skip infrastructure dependencies
    local_output_dir: Optional[str] = None  # Output directory for local mode

    # Additional params (service-specific)
    params: Dict[str, Any] = field(default_factory=dict)


class BaseService(ABC):
    """
    Abstract base class for all inference services.

    Provides:
    - Common initialization (input handler, db writer)
    - Lifecycle management (start, stop, run)
    - Signal handling for graceful shutdown
    - Abstract methods for service-specific logic

    Subclasses must implement:
    - initialize(): Service-specific initialization (load models, etc.)
    - process_frame(): Process a single frame
    - cleanup(): Service-specific cleanup
    """

    def __init__(self, config: ServiceConfig):
        self.config = config
        self._running = False
        self._start_time: Optional[datetime] = None

        # Will be initialized in setup()
        self._input_handler: Optional[FrameInputHandler] = None
        self._db_writer: Optional[BaseWriter] = None

        # Statistics
        self._frames_processed = 0
        self._total_processing_time_ms = 0
        self._error: Optional[str] = None  # Set if service failed

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def setup(self) -> bool:
        """
        Set up common service components.

        Returns:
            True if setup successful
        """
        try:
            # Set up input handler (using new abstraction layer)
            self._input_handler = FrameInputHandler(
                input_source=self.config.input_source,
                input_type=self.config.input_type,
                target_width=self.config.target_width,
                target_height=self.config.target_height,
                start_frame=self.config.start_frame,
                start_segment=self.config.start_segment,
                frame_skip=self.config.frame_skip,
            )

            if not self._input_handler.initialize():
                logger.error("Failed to initialize input handler")
                return False

            # Set up writer (DB or local file based on config)
            self._db_writer = create_writer(
                table_name=self.config.db_table_name,
                local_mode=self.config.local_mode,
                local_output_dir=self.config.local_output_dir,
                match_id=self.config.match_id,
                service_id=self.config.service_id,
            )

            if self.config.local_mode or self.config.local_output_dir:
                logger.info(f"Using LOCAL FILE output: {self.config.local_output_dir}")
            else:
                logger.info(f"Using DynamoDB output: {self.config.db_table_name}")

            # Call service-specific initialization
            if not self.initialize():
                logger.error("Failed service-specific initialization")
                return False

            logger.info(f"Service {self.config.service_id} setup complete")
            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Service-specific initialization.

        Called after common setup. Subclasses should:
        - Load models
        - Initialize service-specific resources
        - Validate configuration

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    def process_frame(self, frame_packet: FrameInputPacket) -> Optional[Dict[str, Any]]:
        """
        Process a single frame.

        Args:
            frame_packet: Frame data and metadata from input handler

        Returns:
            Result dict to write to DB, or None to skip writing
        """
        pass

    def process_batch(
        self,
        frame_packets: List[FrameInputPacket]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process a batch of frames.

        Override this method for efficient batched inference.
        Default implementation calls process_frame for each frame (no speedup).

        Args:
            frame_packets: List of frame packets to process

        Returns:
            List of result dicts (same length as input), None entries skip writing
        """
        # Default: call process_frame for each (no batching benefit)
        return [self.process_frame(pkt) for pkt in frame_packets]

    @property
    def supports_batching(self) -> bool:
        """
        Whether this service supports efficient batched processing.

        Override to return True if process_batch is implemented efficiently.
        """
        return False

    @property
    def supports_multi_worker(self) -> bool:
        """
        Whether this service supports multi-worker parallel processing.

        Override to return True if worker_init and worker_process_batch are implemented.
        """
        return False

    def get_model_path(self) -> str:
        """
        Get the path to the model file.

        Override in subclass to return the model path for worker initialization.
        Required for multi-worker mode.
        """
        raise NotImplementedError("Subclass must implement get_model_path for multi-worker mode")

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration for worker initialization.

        Override in subclass to return config dict for worker_init.
        Required for multi-worker mode.
        """
        raise NotImplementedError("Subclass must implement get_model_config for multi-worker mode")

    @staticmethod
    def worker_init(
        device: str,
        model_path: str,
        **config
    ) -> Any:
        """
        Initialize model in worker process.

        Called once when worker starts. Returns the model object.
        Override in subclass for multi-worker support.

        Args:
            device: Device to load model on (e.g., "cuda:0")
            model_path: Path to model file
            **config: Additional model configuration

        Returns:
            Initialized model object
        """
        raise NotImplementedError("Subclass must implement worker_init for multi-worker mode")

    @staticmethod
    def worker_process_batch(
        model: Any,
        frames: List,
        metadata: List[Dict[str, Any]],
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process a batch of frames in worker process.

        Override in subclass for multi-worker support.

        Args:
            model: Model object from worker_init
            frames: List of numpy arrays (frames)
            metadata: List of metadata dicts for each frame

        Returns:
            List of result dicts (same length as frames)
        """
        raise NotImplementedError("Subclass must implement worker_process_batch for multi-worker mode")

    @abstractmethod
    def cleanup(self):
        """
        Service-specific cleanup.

        Called during shutdown. Subclasses should:
        - Release model resources
        - Close any open connections
        - Clean up temporary files
        """
        pass

    def run(self):
        """
        Main service loop.

        Processes frames from the input handler until stopped or stream ends.
        Chooses mode based on config:
        - num_workers > 1: Distributed mode (multi-worker parallel processing)
        - Otherwise: Local mode (in-process with optional batching)
        """
        if not self.setup():
            logger.error("Setup failed, cannot run service")
            return

        self._running = True
        self._start_time = datetime.utcnow()
        self._setup_signal_handlers()

        num_workers = self.config.num_workers
        batch_size = self.config.inference_batch_size

        if num_workers > 1 and self.supports_multi_worker:
            logger.info(
                f"Service {self.config.service_id} starting "
                f"(distributed mode, workers={num_workers}, batch_size={batch_size})"
            )
            self._run_distributed(num_workers, batch_size)
        else:
            if num_workers > 1 and not self.supports_multi_worker:
                logger.warning(
                    f"num_workers={num_workers} requested but service doesn't support "
                    f"multi-worker mode. Falling back to local mode."
                )
            if batch_size > 1 and not self.supports_batching:
                logger.warning(
                    f"batch_size={batch_size} requested but service doesn't support "
                    f"efficient batching. Will process frames individually."
                )
            mode_desc = f"batch_size={batch_size}" if batch_size > 1 else "single-frame"
            logger.info(f"Service {self.config.service_id} starting (local mode, {mode_desc})")
            self._run_local(batch_size)

    def _run_local(self, batch_size: int):
        """
        Run loop with in-process execution.

        Handles both single-frame (batch_size=1) and batched (batch_size>1) processing
        in a unified code path. When batch_size=1, uses process_frame() directly for
        optimal performance. When batch_size>1, uses process_batch() for GPU efficiency.
        """
        accumulator = BatchAccumulator(batch_size=batch_size)

        try:
            for batch in accumulator.batches(self._input_handler.iterate()):
                if not self._running:
                    logger.info("Service stopped by signal")
                    break

                start_time = datetime.utcnow()

                # Use optimal code path based on batch size
                if batch_size == 1:
                    # Single-frame: direct call, no list overhead
                    result = self.process_frame(batch.items[0])
                    results = [result]
                else:
                    # Batched: process all frames together for GPU efficiency
                    results = self.process_batch(batch.items)

                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                per_frame_time = processing_time / len(batch)

                # Write results to DB
                for frame_packet, result in zip(batch.items, results):
                    if result is not None:
                        self._write_result(result, frame_packet, per_frame_time)

                # Update statistics
                self._frames_processed += len(batch)
                self._total_processing_time_ms += processing_time

                # Log progress
                self._log_progress()

        except Exception as e:
            logger.error(f"Error in local service loop: {e}")
            self._error = str(e)
        finally:
            self.shutdown()

    def _run_distributed(self, num_workers: int, batch_size: int):
        """
        Run with multiple worker processes for true parallel inference.

        Architecture:
        - Main thread: Distributes frames to worker queues
        - Worker processes: Run inference in parallel
        - Collector thread: Gathers results from output queue

        Args:
            num_workers: Number of worker processes
            batch_size: Frames per batch
        """
        from threading import Thread
        from core.worker_pool import WorkerPool
        from core.frame_distributor import FrameDistributor

        # Get model config for workers
        model_path = self.get_model_path()
        model_config = self.get_model_config()

        # Build worker configs - distribute across GPUs if multiple
        base_device = self.config.device
        worker_configs = []
        for i in range(num_workers):
            # If device is cuda:X, can distribute across multiple GPUs
            device = base_device
            if base_device.startswith("cuda:"):
                # For now, use same device - could be extended to multi-GPU
                device = base_device

            worker_configs.append({
                "device": device,
                "model_path": model_path,
                **model_config,
            })

        # Calculate queue size
        queue_size = self.config.worker_queue_size
        if queue_size == 0:
            queue_size = num_workers * 4

        # Create worker pool
        pool = WorkerPool(
            num_workers=num_workers,
            worker_init_fn=self.__class__.worker_init,
            worker_process_fn=self.__class__.worker_process_batch,
            worker_configs=worker_configs,
            queue_size=queue_size,
        )

        # Start workers
        pool.start()

        # Create frame distributor
        distributor = FrameDistributor(
            frame_source=self._input_handler.iterate(),
            worker_pool=pool,
            batch_size=batch_size,
        )

        # Start result collector thread
        collector_stop = False
        collector_error = None

        def collect_results():
            nonlocal collector_error
            try:
                for result in pool.results(timeout=1.0):
                    if not self._running:
                        break

                    if result.error:
                        logger.error(f"Worker {result.worker_id} error: {result.error}")
                        continue

                    # Write results to DB
                    per_frame_time = result.processing_time_ms / len(result.results)
                    for i, (frame_result, meta) in enumerate(zip(result.results, result.metadata if hasattr(result, 'metadata') else [{}] * len(result.results))):
                        if frame_result is not None:
                            # Add common fields
                            frame_result["pk"] = f"{self.config.match_id}#{self.config.service_id}"
                            frame_result["sk"] = meta.get("sequence_number", result.batch_id * batch_size + i)
                            frame_result["match_id"] = self.config.match_id
                            frame_result["service_id"] = self.config.service_id
                            frame_result["frame_number"] = meta.get("sequence_number", result.batch_id * batch_size + i)
                            frame_result["timestamp_ms"] = meta.get("timestamp_ms", 0)
                            frame_result["processing_time_ms"] = int(per_frame_time)

                            self._db_writer.queue_item(frame_result)

                    # Update statistics
                    self._frames_processed += len(result.results)
                    self._total_processing_time_ms += result.processing_time_ms

                    # Log progress
                    self._log_progress()

            except Exception as e:
                collector_error = str(e)
                logger.error(f"Error in result collector: {e}")

        collector_thread = Thread(target=collect_results, daemon=True)
        collector_thread.start()

        try:
            # Start distribution
            distributor.start()

            # Wait for distribution to complete
            while not distributor.is_done:
                if not self._running:
                    logger.info("Service stopped by signal")
                    distributor.stop()
                    break
                distributor.wait(timeout=1.0)

            # Wait for all results to be collected
            logger.info("Distribution complete, waiting for results...")

            # Give workers time to finish
            import time
            timeout = 30.0
            start = time.time()
            while pool._batches_completed < pool._batches_submitted:
                if time.time() - start > timeout:
                    logger.warning("Timeout waiting for worker results")
                    break
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in distributed service loop: {e}")
            self._error = str(e)

        finally:
            # Stop everything
            distributor.stop()
            pool.stop(timeout=10.0)
            collector_thread.join(timeout=5.0)

            if collector_error:
                self._error = collector_error

            # Log distributor stats
            stats = distributor.stats
            logger.info(
                f"Distributor: {stats.frames_distributed} frames, "
                f"{stats.batches_distributed} batches, "
                f"{stats.frames_per_second:.1f} FPS"
            )

            self.shutdown()

    def _write_result(
        self,
        result: Dict[str, Any],
        frame_packet: FrameInputPacket,
        processing_time: float
    ):
        """Write a result to the database with common fields."""
        result["pk"] = f"{self.config.match_id}#{self.config.service_id}"
        result["sk"] = frame_packet.sequence_number
        result["match_id"] = self.config.match_id
        result["service_id"] = self.config.service_id
        result["frame_number"] = frame_packet.sequence_number
        result["timestamp_ms"] = frame_packet.timestamp_ms
        result["processing_time_ms"] = int(processing_time)

        self._db_writer.queue_item(result)

    def _log_progress(self):
        """Log processing progress periodically."""
        if self._frames_processed % 500 == 0:
            avg_time = self._total_processing_time_ms / self._frames_processed
            logger.info(
                f"Processed {self._frames_processed} frames, "
                f"avg processing time: {avg_time:.1f}ms"
            )

    def stop(self):
        """Signal the service to stop"""
        logger.info(f"Stopping service {self.config.service_id}")
        self._running = False

    def shutdown(self):
        """Clean up and shut down the service"""
        logger.info(f"Shutting down service {self.config.service_id}")

        # Calculate final statistics
        elapsed = 0.0
        fps = 0.0
        if self._start_time:
            elapsed = (datetime.utcnow() - self._start_time).total_seconds()
            fps = self._frames_processed / elapsed if elapsed > 0 else 0

        # Write service completion status to DB (before stopping writer)
        if self._db_writer:
            status = "FAILED" if self._error else "COMPLETED"
            status_item = {
                "pk": f"{self.config.match_id}#service_status",
                "sk": self.config.service_id,
                "match_id": self.config.match_id,
                "service_id": self.config.service_id,
                "status": status,
                "frames_processed": self._frames_processed,
                "elapsed_seconds": round(elapsed, 1),
                "avg_fps": round(fps, 1),
                "error": self._error,
                "ended_at": datetime.utcnow().isoformat() + "Z",
            }
            try:
                self._db_writer.write_item(status_item, immediate=True)
                logger.info(f"Service status written: {status}")
            except Exception as e:
                logger.warning(f"Failed to write service status: {e}")

        # Stop input handler
        if self._input_handler:
            self._input_handler.stop()

        # Stop DB writer (will flush remaining items)
        if self._db_writer:
            self._db_writer.stop()

        # Service-specific cleanup
        self.cleanup()

        # Log final statistics
        logger.info(
            f"Service {self.config.service_id} finished: "
            f"{self._frames_processed} frames in {elapsed:.1f}s ({fps:.1f} FPS)"
        )

    def _setup_signal_handlers(self):
        """Set up handlers for graceful shutdown signals"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            self.stop()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if service is running"""
        return self._running

    @property
    def frames_processed(self) -> int:
        """Number of frames processed"""
        return self._frames_processed

    @property
    def input_handler(self) -> Optional[FrameInputHandler]:
        """Access to input handler (for advanced use)"""
        return self._input_handler

    @property
    def db_writer(self) -> Optional[BaseWriter]:
        """Access to DB writer (for advanced use)"""
        return self._db_writer

    @property
    def fps(self) -> Optional[float]:
        """Get FPS from input handler metadata"""
        return self._input_handler.fps if self._input_handler else None
