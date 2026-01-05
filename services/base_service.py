"""
Base Service

Abstract base class for all inference services.
Defines the common lifecycle and interface that all services must implement.

Configuration Hierarchy:
- config/schemas.py::ServiceConfig - "What to run" (from MongoDB/YAML game config)
- ServiceRunConfig (here) - "How to run it" (runtime config with stream, match ID, etc.)

The orchestrator creates ServiceRunConfig by combining GameConfig + MatchConfig.
"""

import os
import logging
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

from core.frame_provider import FrameProvider, FrameProviderConfig, FramePacket, StreamType
from db.dynamo import DynamoDBWriter, DynamoDBWriterConfig, LocalFileWriter, create_writer

logger = logging.getLogger(__name__)


@dataclass
class ServiceRunConfig:
    """
    Runtime configuration for running a service.

    This is the "how to run" config created by combining:
    - GameConfig.services[n] (what service type, params)
    - MatchConfig (stream URL, match ID)
    - InferenceSettings (resolution, frame skip)

    Extended by specific configs (ODServiceConfig, CameraViewServiceConfig)
    """
    # Identifiers
    match_id: str
    service_id: str  # Unique ID for this service instance (e.g., "od_football_v2")

    # Stream configuration
    stream_url: str
    stream_type: str = "hls"

    # Processing settings
    target_width: Optional[int] = None  # None = use source resolution
    target_height: Optional[int] = None
    frame_skip: int = 1  # Process every Nth frame
    start_frame: int = 0  # For resume support
    start_segment: int = 1  # For HLS resume support

    # Device settings
    device: str = "cpu"  # "cpu", "cuda:0", "cuda:1", etc.

    # Database settings
    db_table_name: str = "inference_results"
    db_batch_size: int = 12
    db_flush_interval_ms: int = 250

    # Local testing mode
    local_output_dir: Optional[str] = None  # If set, write to files instead of DB

    # Additional params (service-specific)
    params: Dict[str, Any] = field(default_factory=dict)


# Backward compatibility alias
ServiceConfig = ServiceRunConfig


class BaseService(ABC):
    """
    Abstract base class for all inference services.

    Provides:
    - Common initialization (frame provider, db writer)
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
        self._frame_provider: Optional[FrameProvider] = None
        self._db_writer: Optional[DynamoDBWriter] = None

        # Statistics
        self._frames_processed = 0
        self._total_processing_time_ms = 0

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
            # Validate start_frame and start_segment alignment
            # IMPORTANT: These must come from HLS metadata for proper alignment
            # Passing start_frame without correct start_segment will cause frame misalignment!
            if self.config.start_frame > 0 and self.config.start_segment <= 1:
                logger.warning(
                    f"start_frame={self.config.start_frame} but start_segment={self.config.start_segment}. "
                    f"Frame numbers may not align correctly! "
                    f"For proper resume, both values must come from HLS metadata lookup."
                )

            # Set up frame provider
            frame_config = FrameProviderConfig(
                stream_url=self.config.stream_url,
                stream_type=StreamType(self.config.stream_type),
                target_width=self.config.target_width,
                target_height=self.config.target_height,
                start_frame=self.config.start_frame,
                start_segment=self.config.start_segment,
                frame_skip=self.config.frame_skip,
            )
            self._frame_provider = FrameProvider(frame_config)

            if not self._frame_provider.initialize():
                logger.error("Failed to initialize frame provider")
                return False

            # Set up writer (DB or local file based on config)
            self._db_writer = create_writer(
                table_name=self.config.db_table_name,
                use_background=True,
                local_output_dir=self.config.local_output_dir,
                match_id=self.config.match_id,
                service_id=self.config.service_id,
            )

            if self.config.local_output_dir:
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
    def process_frame(self, frame_packet: FramePacket) -> Optional[Dict[str, Any]]:
        """
        Process a single frame.

        Args:
            frame_packet: Frame data and metadata

        Returns:
            Result dict to write to DB, or None to skip writing
        """
        pass

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

        Processes frames from the stream until stopped or stream ends.
        """
        if not self.setup():
            logger.error("Setup failed, cannot run service")
            return

        self._running = True
        self._start_time = datetime.utcnow()
        self._setup_signal_handlers()

        logger.info(f"Service {self.config.service_id} starting")

        try:
            for frame_packet in self._frame_provider.frames():
                if not self._running:
                    logger.info("Service stopped by signal")
                    break

                # Process frame
                start_time = datetime.utcnow()
                result = self.process_frame(frame_packet)
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                # Write result to DB if provided
                if result is not None:
                    # Add common fields
                    result["pk"] = f"{self.config.match_id}#{self.config.service_id}"
                    result["sk"] = frame_packet.frame_number
                    result["match_id"] = self.config.match_id
                    result["service_id"] = self.config.service_id
                    result["frame_number"] = frame_packet.frame_number
                    result["timestamp_ms"] = frame_packet.timestamp_ms
                    result["processing_time_ms"] = int(processing_time)

                    self._db_writer.queue_item(result)

                # Update statistics
                self._frames_processed += 1
                self._total_processing_time_ms += processing_time

                # Log progress
                if self._frames_processed % 500 == 0:
                    avg_time = self._total_processing_time_ms / self._frames_processed
                    logger.info(
                        f"Processed {self._frames_processed} frames, "
                        f"avg processing time: {avg_time:.1f}ms"
                    )

        except Exception as e:
            logger.error(f"Error in service loop: {e}")
        finally:
            self.shutdown()

    def stop(self):
        """Signal the service to stop"""
        logger.info(f"Stopping service {self.config.service_id}")
        self._running = False

    def shutdown(self):
        """Clean up and shut down the service"""
        logger.info(f"Shutting down service {self.config.service_id}")

        # Stop frame provider
        if self._frame_provider:
            self._frame_provider.stop()

        # Stop DB writer (will flush remaining items)
        if self._db_writer:
            self._db_writer.stop()

        # Service-specific cleanup
        self.cleanup()

        # Log final statistics
        if self._start_time:
            elapsed = (datetime.utcnow() - self._start_time).total_seconds()
            fps = self._frames_processed / elapsed if elapsed > 0 else 0
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
    def frame_provider(self) -> Optional[FrameProvider]:
        """Access to frame provider (for advanced use)"""
        return self._frame_provider

    @property
    def db_writer(self) -> Optional[DynamoDBWriter]:
        """Access to DB writer (for advanced use)"""
        return self._db_writer
