"""
Match Orchestrator

Manages the lifecycle of all services for a single match.
- Loads and freezes configuration
- Spawns services as subprocesses
- Monitors service health
- Handles graceful shutdown
"""

import os
import sys
import json
import time
import signal
import logging
import subprocess
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Delay after starting HLS metadata service before spawning other services
HLS_METADATA_STARTUP_DELAY = 30  # seconds


@dataclass
class ServiceProcess:
    """Represents a spawned service process"""
    service_id: str
    service_type: str
    process: subprocess.Popen
    device: str
    started_at: datetime


class MatchOrchestrator:
    """
    Orchestrates all services for a single match.

    Responsibilities:
    - Load game configuration (from MongoDB or YAML)
    - Freeze configuration at match start
    - Spawn service processes based on config
    - Monitor service health
    - Handle graceful shutdown

    Usage:
        orchestrator = MatchOrchestrator(
            match_id="match_123",
            game_id="6622504e845d0e572cddc306",
            stream_url="https://example.com/stream.m3u8"
        )
        orchestrator.start()  # Spawns all services
        orchestrator.wait()   # Wait for completion
    """

    def __init__(
        self,
        match_id: str,
        stream_url: str,
        game_id: Optional[str] = None,
        config_path: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        local_output_dir: Optional[str] = None,
        enable_resume: bool = True
    ):
        """
        Initialize orchestrator.

        Args:
            match_id: Unique match identifier
            stream_url: Stream URL for this match
            game_id: Game ID to load config from MongoDB
            config_path: Path to YAML config file (alternative to game_id)
            mongo_uri: MongoDB URI (required if using game_id)
            local_output_dir: If set, write to local files instead of DynamoDB (for testing)
            enable_resume: If True, query HLS metadata for resume position
        """
        self.match_id = match_id
        self.stream_url = stream_url
        self.game_id = game_id
        self.config_path = config_path
        self.mongo_uri = mongo_uri
        self.local_output_dir = local_output_dir
        self.enable_resume = enable_resume

        self._config: Optional[Dict[str, Any]] = None
        self._processes: List[ServiceProcess] = []
        self._running = False
        self._resume_position: Dict[str, int] = {"segment_number": 1, "frame_number": 0}

    def _load_config(self) -> Dict[str, Any]:
        """Load and return game configuration"""
        if self.config_path:
            # Load from YAML
            from config.loader import ConfigLoader
            config = ConfigLoader.load_from_yaml(self.config_path)
            return config.model_dump()

        elif self.game_id and self.mongo_uri:
            # Load from MongoDB
            from config.loader import ConfigLoader
            loader = ConfigLoader(mongo_uri=self.mongo_uri)
            config = loader.load_game_config(self.game_id)
            if config is None:
                raise ValueError(f"Game config not found for game_id: {self.game_id}")
            return config.model_dump()

        else:
            raise ValueError("Either config_path or (game_id + mongo_uri) must be provided")

    def start(self):
        """
        Start all services for this match.

        Service startup order:
        1. HLS Metadata Service (extracts frame metadata for resume)
        2. Wait for HLS_METADATA_STARTUP_DELAY seconds
        3. Query resume position (if enable_resume=True)
        4. Spawn OD and Camera View services with resume position
        """
        # Load and freeze configuration
        logger.info(f"Loading configuration for match {self.match_id}")
        self._config = self._load_config()

        # Set up signal handlers
        self._setup_signal_handlers()

        self._running = True
        logger.info(f"Starting services for match {self.match_id}")

        # Get inference settings
        inference_settings = self._config.get("inference_settings", {})

        # Step 1: Start HLS Metadata Service FIRST (if enabled)
        hls_metadata_enabled = self._is_hls_metadata_enabled()
        if hls_metadata_enabled:
            self._spawn_hls_metadata_service()

            # Step 2: Wait for metadata service to collect some data
            logger.info(f"Waiting {HLS_METADATA_STARTUP_DELAY}s for HLS metadata collection...")
            time.sleep(HLS_METADATA_STARTUP_DELAY)

        # Step 3: Get resume position (if enabled and not local testing)
        if self.enable_resume and not self.local_output_dir:
            self._resume_position = self._get_resume_position()
            logger.info(f"Resume position: {self._resume_position}")

        # Step 4: Spawn remaining services
        for service_config in self._config.get("services", []):
            if not service_config.get("enabled", True):
                logger.info(f"Skipping disabled service: {service_config.get('service_type')}")
                continue

            service_type = service_config.get("service_type")

            if service_type == "object_detection":
                self._spawn_od_services(service_config, inference_settings)
            elif service_type == "camera_view":
                self._spawn_camera_view_service(service_config, inference_settings)
            elif service_type == "hls_metadata":
                # Already started above
                pass
            else:
                logger.warning(f"Unknown service type: {service_type}")

        logger.info(f"Started {len(self._processes)} services")

    def _is_hls_metadata_enabled(self) -> bool:
        """Check if HLS metadata service is enabled in config"""
        for service_config in self._config.get("services", []):
            if service_config.get("service_type") == "hls_metadata":
                return service_config.get("enabled", True)
        # Default: enable HLS metadata for HLS streams
        return self.stream_url.endswith(".m3u8")

    def _spawn_hls_metadata_service(self):
        """Spawn HLS Metadata Service"""
        cmd = [
            sys.executable, "-m", "services.hls_metadata_service",
            "--match-id", self.match_id,
            "--stream-url", self.stream_url,
        ]

        if self.local_output_dir:
            cmd.extend(["--local-output", self.local_output_dir])

        logger.info("Spawning HLS Metadata Service")
        process = subprocess.Popen(cmd)

        self._processes.append(ServiceProcess(
            service_id="hls_metadata",
            service_type="hls_metadata",
            process=process,
            device="cpu",
            started_at=datetime.utcnow()
        ))

    def _get_resume_position(self) -> Dict[str, int]:
        """Query HLS metadata for resume position"""
        try:
            from services.hls_metadata_service.service import get_resume_position
            return get_resume_position(
                match_id=self.match_id,
                stream_url=self.stream_url
            )
        except Exception as e:
            logger.warning(f"Failed to get resume position: {e}")
            return {"segment_number": 1, "frame_number": 0}

    def _spawn_od_services(self, service_config: Dict, inference_settings: Dict):
        """Spawn Object Detection services (one per model)"""
        params = service_config.get("params", {})
        model_ids = params.get("model_ids", [])
        device = service_config.get("device", "cuda:0")

        # Get device assignment if specified
        device_assignment = params.get("device_assignment", {})

        for model_id in model_ids:
            # Find model config
            model_config = None
            for m in self._config.get("models", []):
                if m.get("model_id") == model_id:
                    model_config = m
                    break

            if model_config is None:
                logger.error(f"Model config not found for model_id: {model_id}")
                continue

            # Get device for this model
            model_device = device_assignment.get(model_id, device)

            # Build command
            cmd = self._build_od_command(model_config, model_device, inference_settings)

            # Spawn process
            logger.info(f"Spawning OD service: model={model_id}, device={model_device}")
            process = subprocess.Popen(cmd)

            self._processes.append(ServiceProcess(
                service_id=f"od_{model_id}",
                service_type="object_detection",
                process=process,
                device=model_device,
                started_at=datetime.utcnow()
            ))

    def _build_od_command(
        self,
        model_config: Dict,
        device: str,
        inference_settings: Dict
    ) -> List[str]:
        """Build command to run OD service"""
        model_params = model_config.get("params", {})

        # Build class mapping JSON
        class_mapping = {}
        for cm in model_config.get("class_mapping", []):
            class_mapping[cm["model_class_id"]] = cm["universal_class_name"]

        cmd = [
            sys.executable, "-m", "services.od_service",
            "--match-id", self.match_id,
            "--stream-url", self.stream_url,
            "--model-id", model_config.get("model_id"),
            "--device", device,
            "--confidence", str(model_params.get("confidence_threshold", 0.5)),
        ]

        # Add model URL/path
        if model_config.get("model_url"):
            cmd.extend(["--model-url", model_config["model_url"]])

        # Add class mapping
        if class_mapping:
            cmd.extend(["--class-mapping", json.dumps(class_mapping)])

        # Add classes to predict
        classes = model_config.get("classes_to_predict", [])
        if classes:
            cmd.extend(["--classes", ",".join(str(c) for c in classes)])

        # Add resolution
        resolution = inference_settings.get("processing_resolution", [1280, 720])
        cmd.extend(["--width", str(resolution[0])])
        cmd.extend(["--height", str(resolution[1])])

        # Add resume position
        cmd.extend(["--start-frame", str(self._resume_position.get("frame_number", 0))])
        cmd.extend(["--start-segment", str(self._resume_position.get("segment_number", 1))])

        # Add local output directory (for testing)
        if self.local_output_dir:
            cmd.extend(["--local-output", self.local_output_dir])

        return cmd

    def _spawn_camera_view_service(self, service_config: Dict, inference_settings: Dict):
        """Spawn Camera View service"""
        params = service_config.get("params", {})
        resolution = inference_settings.get("processing_resolution", [1280, 720])
        scale = params.get("resolution_scale", 0.5)

        cmd = [
            sys.executable, "-m", "services.camera_view_service",
            "--match-id", self.match_id,
            "--stream-url", self.stream_url,
            "--phash-threshold", str(params.get("phash_threshold", 20)),
            "--histogram-threshold", str(params.get("histogram_threshold", 0.90)),
            "--min-frame-gap", str(params.get("min_frame_gap", 25)),
            "--width", str(int(resolution[0] * scale)),
            "--height", str(int(resolution[1] * scale)),
        ]

        # Add resume position
        cmd.extend(["--start-frame", str(self._resume_position.get("frame_number", 0))])
        cmd.extend(["--start-segment", str(self._resume_position.get("segment_number", 1))])

        # Add local output directory (for testing)
        if self.local_output_dir:
            cmd.extend(["--local-output", self.local_output_dir])

        logger.info("Spawning Camera View service")
        process = subprocess.Popen(cmd)

        self._processes.append(ServiceProcess(
            service_id="camera_view",
            service_type="camera_view",
            process=process,
            device="cpu",
            started_at=datetime.utcnow()
        ))

    def wait(self, timeout: Optional[float] = None):
        """
        Wait for all services to complete.

        Args:
            timeout: Maximum time to wait (None = wait forever)
        """
        logger.info("Waiting for services to complete")

        for sp in self._processes:
            try:
                sp.process.wait(timeout=timeout)
                logger.info(f"Service {sp.service_id} completed")
            except subprocess.TimeoutExpired:
                logger.warning(f"Service {sp.service_id} timed out")

    def stop(self):
        """Stop all running services gracefully"""
        logger.info("Stopping all services")
        self._running = False

        for sp in self._processes:
            if sp.process.poll() is None:  # Still running
                logger.info(f"Terminating service {sp.service_id}")
                sp.process.terminate()

        # Wait for processes to terminate
        for sp in self._processes:
            try:
                sp.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing service {sp.service_id}")
                sp.process.kill()

    def _setup_signal_handlers(self):
        """Set up handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, stopping orchestrator")
            self.stop()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    @property
    def is_running(self) -> bool:
        """Check if any services are still running"""
        return any(sp.process.poll() is None for sp in self._processes)

    @property
    def service_status(self) -> Dict[str, str]:
        """Get status of all services"""
        status = {}
        for sp in self._processes:
            poll = sp.process.poll()
            if poll is None:
                status[sp.service_id] = "running"
            elif poll == 0:
                status[sp.service_id] = "completed"
            else:
                status[sp.service_id] = f"failed (exit code: {poll})"
        return status


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for running orchestrator"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Match Orchestrator")
    parser.add_argument("--match-id", required=True, help="Match identifier")
    parser.add_argument("--stream-url", required=True, help="Stream URL")
    parser.add_argument("--game-id", help="Game ID (for MongoDB config)")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--mongo-uri", help="MongoDB URI")
    parser.add_argument("--local-output", help="Local output directory (for testing without DynamoDB)")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from last position")

    args = parser.parse_args()

    if not args.config and not (args.game_id and args.mongo_uri):
        parser.error("Either --config or (--game-id and --mongo-uri) must be provided")

    orchestrator = MatchOrchestrator(
        match_id=args.match_id,
        stream_url=args.stream_url,
        game_id=args.game_id,
        config_path=args.config,
        mongo_uri=args.mongo_uri,
        local_output_dir=args.local_output,
        enable_resume=not args.no_resume
    )

    orchestrator.start()
    orchestrator.wait()


if __name__ == "__main__":
    main()
