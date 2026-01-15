"""
Match Orchestrator

Manages the lifecycle of all services for a single match.
- Loads configuration from unified YAML
- Spawns services as subprocesses
- Monitors service health
- Handles graceful shutdown

Usage:
    # Unified run config (recommended)
    python -m orchestrator.match_orchestrator --run-config config/run_config.yaml

    # Separate CLI args (legacy)
    python -m orchestrator.match_orchestrator --match-id test --stream-url https://... --config game.yaml
"""

import os
import sys
import json
import time
import signal
import logging
import subprocess
import yaml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


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
    - Load configuration from unified YAML or separate sources
    - Spawn service processes based on config
    - Monitor service health
    - Handle graceful shutdown
    """

    def __init__(
        self,
        match_id: str,
        stream_url: str,
        stream_type: str = "hls",
        config_path: Optional[str] = None,
        game_id: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        local_output_dir: Optional[str] = None,
        enable_resume: bool = True,
        hls_head_start_seconds: int = 30,
    ):
        """
        Initialize orchestrator.

        Args:
            match_id: Unique match identifier
            stream_url: Stream URL for this match
            stream_type: Type of stream (hls, mp4, file)
            config_path: Path to YAML config file
            game_id: Game ID to load config from MongoDB
            mongo_uri: MongoDB URI (required if using game_id)
            local_output_dir: If set, write to local files instead of DynamoDB
            enable_resume: If True, query HLS metadata for resume position
            hls_head_start_seconds: Seconds to wait after HLS metadata starts
        """
        self.match_id = match_id
        self.stream_url = stream_url
        self.stream_type = stream_type
        self.config_path = config_path
        self.game_id = game_id
        self.mongo_uri = mongo_uri
        self.local_output_dir = local_output_dir
        self.enable_resume = enable_resume
        self.hls_head_start_seconds = hls_head_start_seconds

        self._config: Optional[Dict[str, Any]] = None
        self._processes: List[ServiceProcess] = []
        self._running = False
        self._resume_position: Dict[str, int] = {"segment_number": 1, "frame_number": 0}

    @classmethod
    def from_run_config(cls, run_config_path: str) -> "MatchOrchestrator":
        """
        Create orchestrator from a unified run config YAML file.

        The run config includes everything in one file:
        - match_id, stream_url, stream_type
        - deployment settings (local vs prod)
        - services configuration
        - models configuration
        - inference settings

        Args:
            run_config_path: Path to unified run config YAML

        Returns:
            Configured MatchOrchestrator instance
        """
        with open(run_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract required runtime settings
        match_id = config.get("match_id")
        stream_url = config.get("stream_url")

        if not match_id:
            raise ValueError("run_config must include 'match_id'")
        if not stream_url:
            raise ValueError("run_config must include 'stream_url'")

        stream_type = config.get("stream_type", "hls")

        # Extract deployment settings
        deployment = config.get("deployment", {})
        local_output_dir = deployment.get("local_output_dir")
        hls_head_start = deployment.get("hls_metadata_head_start_seconds", 30)
        enable_resume = deployment.get("enable_resume", True)

        # Create instance
        instance = cls(
            match_id=match_id,
            stream_url=stream_url,
            stream_type=stream_type,
            config_path=run_config_path,
            local_output_dir=local_output_dir,
            enable_resume=enable_resume,
            hls_head_start_seconds=hls_head_start,
        )

        return instance

    def _load_config(self) -> Dict[str, Any]:
        """Load and return configuration"""
        if self.config_path:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Check if unified run config (has match_id at top level)
            if "match_id" in config:
                logger.info("Loading unified run config format")
                return config

            # Otherwise parse as game template via ConfigLoader
            logger.info("Loading game template format")
            from config.loader import ConfigLoader
            game_config, model_registry = ConfigLoader.load_from_yaml(self.config_path)
            config_dict = game_config.model_dump()
            config_dict["models"] = [m.model_dump() for m in model_registry.models]
            return config_dict

        elif self.game_id and self.mongo_uri:
            from config.loader import ConfigLoader
            loader = ConfigLoader(mongo_uri=self.mongo_uri)
            game_config = loader.load_game_template(self.game_id)
            model_registry = loader.load_model_registry(self.game_id)
            if game_config is None:
                raise ValueError(f"Game config not found: {self.game_id}")
            config_dict = game_config.model_dump()
            config_dict["models"] = [m.model_dump() for m in model_registry.models]
            return config_dict

        else:
            raise ValueError("Either config_path or (game_id + mongo_uri) required")

    def start(self):
        """
        Start all services for this match.

        Service startup order:
        1. HLS Metadata Service (extracts frame metadata)
        2. Wait for hls_head_start_seconds
        3. Query resume position (if enabled)
        4. Spawn remaining services (Camera View, OD, etc.)
        """
        logger.info(f"Loading configuration for match {self.match_id}")
        self._config = self._load_config()

        self._setup_signal_handlers()
        self._running = True

        logger.info(f"Starting services for match {self.match_id}")
        logger.info(f"Stream: {self.stream_url} (type: {self.stream_type})")
        if self.local_output_dir:
            logger.info(f"Output: LOCAL ({self.local_output_dir})")
        else:
            logger.info("Output: DynamoDB")

        inference_settings = self._config.get("inference_settings") or {}

        # Step 1: Start HLS Metadata Service FIRST (if enabled)
        hls_enabled = self._is_service_enabled("hls_metadata")
        if hls_enabled:
            self._spawn_hls_metadata_service()

            # Step 2: Wait for metadata collection
            logger.info(f"Waiting {self.hls_head_start_seconds}s for HLS metadata...")
            time.sleep(self.hls_head_start_seconds)

        # Step 3: Get resume position
        if self.enable_resume and not self.local_output_dir:
            self._resume_position = self._get_resume_position()
            logger.info(f"Resume position: {self._resume_position}")

        # Step 4: Spawn remaining services in order
        for service_config in self._config.get("services", []):
            if not service_config.get("enabled", True):
                continue

            service_type = service_config.get("service_type")

            if service_type == "hls_metadata":
                pass  # Already started
            elif service_type == "camera_view":
                self._spawn_camera_view_service(service_config, inference_settings)
            elif service_type == "object_detection":
                self._spawn_od_services(service_config, inference_settings)
            elif service_type == "segmentation":
                logger.warning("Segmentation service not yet implemented")
            else:
                logger.warning(f"Unknown service type: {service_type}")

        logger.info(f"Started {len(self._processes)} services")

    def _is_service_enabled(self, service_type: str) -> bool:
        """Check if a service type is enabled in config"""
        for svc in self._config.get("services", []):
            if svc.get("service_type") == service_type:
                return svc.get("enabled", True)
        # Default: enable HLS metadata for HLS streams
        if service_type == "hls_metadata":
            return self.stream_type == "hls" or self.stream_url.endswith(".m3u8")
        return False

    def _get_service_config(self, service_type: str) -> Dict[str, Any]:
        """Get config for a service type"""
        for svc in self._config.get("services", []):
            if svc.get("service_type") == service_type:
                return svc
        return {}

    def _spawn_hls_metadata_service(self):
        """Spawn HLS Metadata Service"""
        service_config = self._get_service_config("hls_metadata")

        cmd = [
            sys.executable, "-m", "services.hls_metadata_service",
            "--match-id", self.match_id,
            "--stream-url", self.stream_url,
        ]

        # Add service-specific settings
        if service_config.get("poll_interval_seconds"):
            cmd.extend(["--poll-interval", str(service_config["poll_interval_seconds"])])
        if service_config.get("timeout_no_segment_seconds"):
            cmd.extend(["--timeout", str(service_config["timeout_no_segment_seconds"])])
        if service_config.get("resolution_preference"):
            cmd.extend(["--resolution-preference", service_config["resolution_preference"]])
        if service_config.get("db_table_name"):
            cmd.extend(["--db-table", service_config["db_table_name"]])

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

    def _spawn_camera_view_service(self, service_config: Dict, inference_settings: Dict):
        """Spawn Camera View service"""
        cmd = [
            sys.executable, "-m", "services.camera_view_service",
            "--match-id", self.match_id,
            "--stream-url", self.stream_url,
            "--input-type", self.stream_type,
            "--phash-threshold", str(service_config.get("phash_threshold", 20)),
            "--histogram-threshold", str(service_config.get("histogram_threshold", 0.90)),
            "--min-frame-gap", str(service_config.get("min_frame_gap", 25)),
            "--start-frame", str(self._resume_position.get("frame_number", 0)),
            "--start-segment", str(self._resume_position.get("segment_number", 1)),
        ]

        # Resolution is optional - only add if specified
        resolution = inference_settings.get("processing_resolution")
        scale = service_config.get("resolution_scale", 0.5)
        if resolution:
            cmd.extend(["--width", str(int(resolution[0] * scale))])
            cmd.extend(["--height", str(int(resolution[1] * scale))])

        if service_config.get("db_table_name"):
            cmd.extend(["--db-table", service_config["db_table_name"]])
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

    def _spawn_od_services(self, service_config: Dict, inference_settings: Dict):
        """Spawn Object Detection services (one per model)"""
        # Support both model_ids (direct) and model_roles (via assignments)
        model_ids = service_config.get("model_ids", [])
        model_roles = service_config.get("model_roles", [])
        default_device = service_config.get("device", "cpu")

        # If model_roles specified, look up model_ids from model_assignments
        if model_roles and not model_ids:
            model_ids = self._get_model_ids_for_roles(model_roles)

        if not model_ids:
            logger.warning("No models specified for OD service (check model_ids or model_roles)")
            return

        # Check CUDA availability once
        cuda_available = self._check_cuda_available()

        for model_id in model_ids:
            model_config = self._get_model_config(model_id)
            if model_config is None:
                logger.error(f"Model not found: {model_id}")
                continue

            # Determine device
            resources = model_config.get("resources", {})
            if resources.get("requires_gpu", False) and cuda_available:
                device = "cuda:0"
            else:
                if resources.get("requires_gpu", False) and not cuda_available:
                    logger.warning(f"Model {model_id} requires GPU but CUDA unavailable, using CPU")
                device = default_device

            cmd = self._build_od_command(model_config, device, service_config, inference_settings)

            logger.info(f"Spawning OD service: model={model_id}, device={device}")
            process = subprocess.Popen(cmd)

            self._processes.append(ServiceProcess(
                service_id=f"od_{model_id}",
                service_type="object_detection",
                process=process,
                device=device,
                started_at=datetime.utcnow()
            ))

    def _get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model config by ID"""
        for m in self._config.get("models", []):
            if m.get("model_id") == model_id:
                return m
        return None

    def _get_model_ids_for_roles(self, roles: List[str]) -> List[str]:
        """Get model IDs from model_assignments by role"""
        model_ids = []
        for assignment in self._config.get("model_assignments", []):
            if assignment.get("role") in roles:
                model_ids.append(assignment.get("model_id"))
        return model_ids

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available for GPU inference"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _build_od_command(
        self,
        model_config: Dict,
        device: str,
        service_config: Dict,
        inference_settings: Dict
    ) -> List[str]:
        """Build command to run OD service"""
        # Support both new (default_params) and old (params) field names
        model_params = model_config.get("default_params", model_config.get("params", {}))

        # Build class mapping - support both formats
        class_mapping = {}
        mapping_list = model_config.get("default_class_mapping", model_config.get("class_mapping", []))
        for cm in mapping_list:
            class_mapping[cm["model_class_id"]] = cm["universal_class_name"]

        cmd = [
            sys.executable, "-m", "services.od_service",
            "--match-id", self.match_id,
            "--stream-url", self.stream_url,
            "--input-type", self.stream_type,
            "--model-id", model_config.get("model_id"),
            "--device", device,
            "--confidence", str(model_params.get("confidence_threshold", 0.5)),
            "--start-frame", str(self._resume_position.get("frame_number", 0)),
            "--start-segment", str(self._resume_position.get("segment_number", 1)),
        ]

        # Resolution is optional - only add if specified
        resolution = inference_settings.get("processing_resolution")
        target_width = service_config.get("target_width")
        target_height = service_config.get("target_height")
        if target_width:
            cmd.extend(["--width", str(target_width)])
        elif resolution:
            cmd.extend(["--width", str(resolution[0])])
        if target_height:
            cmd.extend(["--height", str(target_height)])
        elif resolution:
            cmd.extend(["--height", str(resolution[1])])

        # Model path/URL
        model_path = model_config.get("model_url") or model_config.get("model_path")
        if model_path:
            cmd.extend(["--model-path", model_path])

        # Class mapping
        if class_mapping:
            cmd.extend(["--class-mapping", json.dumps(class_mapping)])

        # Classes to predict - support both formats
        classes = model_config.get("native_class_ids", model_config.get("classes_to_predict", []))
        if classes:
            cmd.extend(["--classes", ",".join(str(c) for c in classes)])

        # DB table
        if service_config.get("db_table_name"):
            cmd.extend(["--db-table", service_config["db_table_name"]])

        # Local output
        if self.local_output_dir:
            cmd.extend(["--local-output", self.local_output_dir])

        return cmd

    def wait(self, timeout: Optional[float] = None):
        """Wait for all services to complete or until stopped"""
        logger.info("Waiting for services to complete (Ctrl+C to stop)")
        try:
            while self._running and self.is_running:
                # Check each process with short timeout to allow interrupt
                for sp in self._processes:
                    if sp.process.poll() is not None:
                        continue
                    try:
                        sp.process.wait(timeout=0.5)
                        logger.info(f"Service {sp.service_id} completed")
                    except subprocess.TimeoutExpired:
                        pass
                # Check if all done
                if not self.is_running:
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted, stopping services...")
            self.stop()

    def stop(self):
        """Stop all running services gracefully"""
        if not self._running:
            return  # Already stopped
        logger.info("Stopping all services")
        self._running = False

        # Send SIGTERM to all running processes
        for sp in self._processes:
            if sp.process.poll() is None:
                logger.info(f"Terminating {sp.service_id}")
                sp.process.terminate()

        # Wait briefly for graceful shutdown
        for sp in self._processes:
            try:
                sp.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {sp.service_id}")
                sp.process.kill()
                sp.process.wait()  # Reap the process

        logger.info("All services stopped")

    def _setup_signal_handlers(self):
        """Set up handlers for graceful shutdown"""
        def handler(signum, frame):
            logger.info(f"Received signal {signum}, stopping")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

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
                status[sp.service_id] = f"failed (exit={poll})"
        return status


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Match Orchestrator - Run inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unified run config (recommended):
  python -m orchestrator.match_orchestrator --run-config config/run_config.yaml

  # Separate args (legacy):
  python -m orchestrator.match_orchestrator --match-id test --stream-url https://... --config game.yaml
        """
    )

    # Option 1: Unified run config
    parser.add_argument("--run-config", help="Path to unified run config YAML")

    # Option 2: Separate arguments
    parser.add_argument("--match-id", help="Match identifier")
    parser.add_argument("--stream-url", help="Stream URL")
    parser.add_argument("--config", help="Path to game template YAML")
    parser.add_argument("--game-id", help="Game ID (for MongoDB)")
    parser.add_argument("--mongo-uri", help="MongoDB URI")
    parser.add_argument("--local-output", help="Local output directory")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume")

    args = parser.parse_args()

    orchestrator = None
    try:
        if args.run_config:
            # Use unified run config
            logger.info(f"Loading run config: {args.run_config}")
            orchestrator = MatchOrchestrator.from_run_config(args.run_config)
        else:
            # Use separate args
            if not args.match_id:
                parser.error("--match-id required when not using --run-config")
            if not args.stream_url:
                parser.error("--stream-url required when not using --run-config")
            if not args.config and not (args.game_id and args.mongo_uri):
                parser.error("--config or (--game-id + --mongo-uri) required")

            orchestrator = MatchOrchestrator(
                match_id=args.match_id,
                stream_url=args.stream_url,
                config_path=args.config,
                game_id=args.game_id,
                mongo_uri=args.mongo_uri,
                local_output_dir=args.local_output,
                enable_resume=not args.no_resume,
            )

        orchestrator.start()
        orchestrator.wait()
        logger.info("Orchestrator finished")

    except KeyboardInterrupt:
        logger.info("Interrupted")
        if orchestrator:
            orchestrator.stop()
    except Exception as e:
        logger.error(f"Error: {e}")
        if orchestrator:
            orchestrator.stop()
        raise


if __name__ == "__main__":
    main()
