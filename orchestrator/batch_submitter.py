"""
AWS Batch Job Submitter

Submits inference jobs to AWS Batch for distributed processing.
Each service runs as an independent Batch job on its own EC2 instance.

Usage:
    # From Lambda or CLI
    from orchestrator.batch_submitter import submit_match_jobs

    submit_match_jobs(
        match_id="match_123",
        stream_url="https://cdn.example.com/stream.m3u8",
        game_config=config,
        enable_resume=False
    )
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for AWS Batch submission"""
    gpu_job_queue: str = "inference-gpu-queue"
    cpu_job_queue: str = "inference-cpu-queue"

    # Job definitions (must be created in AWS Batch)
    od_job_definition: str = "od-service"
    camera_view_job_definition: str = "camera-view-service"
    segmentation_job_definition: str = "segmentation-service"
    hls_metadata_job_definition: str = "hls-metadata-service"

    # Timing
    hls_metadata_head_start_seconds: int = 30

    @classmethod
    def from_env(cls) -> "BatchConfig":
        """Create config from environment variables"""
        return cls(
            gpu_job_queue=os.getenv("BATCH_GPU_QUEUE", "inference-gpu-queue"),
            cpu_job_queue=os.getenv("BATCH_CPU_QUEUE", "inference-cpu-queue"),
            od_job_definition=os.getenv("BATCH_OD_JOB_DEF", "od-service"),
            camera_view_job_definition=os.getenv("BATCH_CV_JOB_DEF", "camera-view-service"),
            hls_metadata_job_definition=os.getenv("BATCH_HLS_JOB_DEF", "hls-metadata-service"),
        )


def get_resume_position(match_id: str, stream_url: str) -> Dict[str, int]:
    """
    Get resume position from HLS metadata in DynamoDB.

    Returns:
        Dict with "segment_number" and "frame_number", defaults to start if not found
    """
    try:
        from services.hls_metadata_service.service import get_resume_position as _get_resume
        return _get_resume(match_id, stream_url)
    except Exception as e:
        logger.warning(f"Failed to get resume position: {e}, starting from beginning")
        return {"segment_number": 1, "frame_number": 0}


def _get_batch_client():
    """Get boto3 Batch client"""
    import boto3
    return boto3.client(
        "batch",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )


def _submit_job(
    batch_client,
    job_name: str,
    job_queue: str,
    job_definition: str,
    parameters: Dict[str, str],
    depends_on: Optional[List[Dict]] = None
) -> str:
    """
    Submit a single job to AWS Batch.

    Returns:
        Job ID
    """
    submit_args = {
        "jobName": job_name,
        "jobQueue": job_queue,
        "jobDefinition": job_definition,
        "parameters": {k: str(v) for k, v in parameters.items()},
    }

    if depends_on:
        submit_args["dependsOn"] = depends_on

    response = batch_client.submit_job(**submit_args)
    job_id = response["jobId"]

    logger.info(f"Submitted job {job_name}: {job_id}")
    return job_id


def submit_match_jobs(
    match_id: str,
    stream_url: str,
    game_config: Dict[str, Any],
    enable_resume: bool = False,
    batch_config: Optional[BatchConfig] = None,
    wait_for_hls_metadata: bool = True,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Submit all jobs for a match to AWS Batch.

    Args:
        match_id: Unique match identifier
        stream_url: HLS/RTSP/MP4 stream URL
        game_config: Game configuration dict (from MongoDB or YAML)
        enable_resume: If True, query DynamoDB for resume position
        batch_config: AWS Batch configuration
        wait_for_hls_metadata: If True, wait before submitting other jobs
        dry_run: If True, don't actually submit (for testing)

    Returns:
        Dict with job IDs and status
    """
    config = batch_config or BatchConfig.from_env()

    if not dry_run:
        batch_client = _get_batch_client()
    else:
        batch_client = None

    submitted_jobs = []
    result = {
        "match_id": match_id,
        "jobs": [],
        "resume_position": None,
    }

    # Extract settings from game config
    services = game_config.get("services", [])
    models = game_config.get("models", [])
    inference_settings = game_config.get("inference_settings", {})

    # Build model lookup
    model_lookup = {m["model_id"]: m for m in models}

    # 1. Check if HLS metadata service is enabled
    hls_metadata_enabled = any(
        s.get("service_type") == "hls_metadata" and s.get("enabled", True)
        for s in services
    )

    # 2. Submit HLS Metadata job first (if enabled)
    hls_job_id = None
    if hls_metadata_enabled:
        job_name = f"{match_id}_hls_metadata"
        params = {
            "match_id": match_id,
            "stream_url": stream_url,
        }

        if dry_run:
            logger.info(f"[DRY RUN] Would submit: {job_name}")
            hls_job_id = "dry-run-hls-job-id"
        else:
            hls_job_id = _submit_job(
                batch_client,
                job_name=job_name,
                job_queue=config.cpu_job_queue,
                job_definition=config.hls_metadata_job_definition,
                parameters=params,
            )

        result["jobs"].append({
            "job_name": job_name,
            "job_id": hls_job_id,
            "service_type": "hls_metadata",
        })

        # Wait for HLS metadata to start indexing
        if wait_for_hls_metadata and not dry_run:
            logger.info(f"Waiting {config.hls_metadata_head_start_seconds}s for HLS metadata indexing...")
            time.sleep(config.hls_metadata_head_start_seconds)

    # 3. Determine resume position (if enabled)
    start_frame = 0
    start_segment = 1

    if enable_resume:
        resume_pos = get_resume_position(match_id, stream_url)
        start_frame = resume_pos.get("frame_number", 0)
        start_segment = resume_pos.get("segment_number", 1)
        result["resume_position"] = resume_pos
        logger.info(f"Resume position: frame={start_frame}, segment={start_segment}")

    # 4. Submit other services
    for service in services:
        service_type = service.get("service_type")

        # Skip HLS metadata (already submitted) and disabled services
        if service_type == "hls_metadata":
            continue
        if not service.get("enabled", True):
            continue

        # Determine job queue based on device
        device = service.get("device", "cpu")
        job_queue = config.gpu_job_queue if "cuda" in device else config.cpu_job_queue

        # Get job definition
        job_def_map = {
            "object_detection": config.od_job_definition,
            "camera_view": config.camera_view_job_definition,
            "segmentation": config.segmentation_job_definition,
        }
        job_definition = job_def_map.get(service_type, service_type)

        # Build parameters based on service type
        service_params = service.get("params", {})

        if service_type == "object_detection":
            # OD service - one job per model
            model_ids = service_params.get("model_ids", [])

            for model_id in model_ids:
                model_config = model_lookup.get(model_id, {})
                job_name = f"{match_id}_od_{model_id}"

                # Build class mapping JSON
                class_mapping = {}
                for cm in model_config.get("class_mapping", []):
                    class_mapping[cm["model_class_id"]] = cm["universal_class_name"]

                model_params = model_config.get("params", {})

                params = {
                    "match_id": match_id,
                    "stream_url": stream_url,
                    "model_id": model_id,
                    "model_url": model_config.get("model_url", ""),
                    "device": device,
                    "start_frame": start_frame,
                    "start_segment": start_segment,
                    "confidence": model_params.get("confidence_threshold", 0.5),
                    "class_mapping": json.dumps(class_mapping),
                    "classes": ",".join(str(c) for c in model_config.get("classes_to_predict", [])),
                    "width": inference_settings.get("processing_resolution", [1280, 720])[0],
                    "height": inference_settings.get("processing_resolution", [1280, 720])[1],
                }

                if dry_run:
                    logger.info(f"[DRY RUN] Would submit: {job_name}")
                    job_id = f"dry-run-{job_name}"
                else:
                    job_id = _submit_job(
                        batch_client,
                        job_name=job_name,
                        job_queue=job_queue,
                        job_definition=job_definition,
                        parameters=params,
                    )

                result["jobs"].append({
                    "job_name": job_name,
                    "job_id": job_id,
                    "service_type": service_type,
                    "model_id": model_id,
                })

        elif service_type == "camera_view":
            job_name = f"{match_id}_camera_view"

            params = {
                "match_id": match_id,
                "stream_url": stream_url,
                "start_frame": start_frame,
                "start_segment": start_segment,
                "phash_threshold": service_params.get("phash_threshold", 20),
                "histogram_threshold": service_params.get("histogram_threshold", 0.90),
                "min_frame_gap": service_params.get("min_frame_gap", 25),
                "width": int(inference_settings.get("processing_resolution", [1280, 720])[0] *
                           service_params.get("resolution_scale", 0.5)),
                "height": int(inference_settings.get("processing_resolution", [1280, 720])[1] *
                            service_params.get("resolution_scale", 0.5)),
            }

            if dry_run:
                logger.info(f"[DRY RUN] Would submit: {job_name}")
                job_id = f"dry-run-{job_name}"
            else:
                job_id = _submit_job(
                    batch_client,
                    job_name=job_name,
                    job_queue=job_queue,
                    job_definition=job_definition,
                    parameters=params,
                )

            result["jobs"].append({
                "job_name": job_name,
                "job_id": job_id,
                "service_type": service_type,
            })

        else:
            # Generic service submission
            job_name = f"{match_id}_{service_type}"

            params = {
                "match_id": match_id,
                "stream_url": stream_url,
                "start_frame": start_frame,
                "start_segment": start_segment,
                **{k: str(v) for k, v in service_params.items()},
            }

            if dry_run:
                logger.info(f"[DRY RUN] Would submit: {job_name}")
                job_id = f"dry-run-{job_name}"
            else:
                job_id = _submit_job(
                    batch_client,
                    job_name=job_name,
                    job_queue=job_queue,
                    job_definition=job_definition,
                    parameters=params,
                )

            result["jobs"].append({
                "job_name": job_name,
                "job_id": job_id,
                "service_type": service_type,
            })

    logger.info(f"Submitted {len(result['jobs'])} jobs for match {match_id}")
    return result


def check_match_status(match_id: str) -> Dict[str, Any]:
    """
    Check status of all services for a match by querying DynamoDB.

    Returns:
        Dict with service statuses
    """
    import boto3
    from boto3.dynamodb.conditions import Key

    dynamodb = boto3.resource(
        "dynamodb",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
    table = dynamodb.Table(os.getenv("DYNAMODB_TABLE", "inference_results"))

    response = table.query(
        KeyConditionExpression=Key("pk").eq(f"{match_id}#service_status")
    )

    services = {}
    for item in response.get("Items", []):
        service_id = item.get("sk")
        services[service_id] = {
            "status": item.get("status"),
            "frames_processed": item.get("frames_processed", 0),
            "elapsed_seconds": item.get("elapsed_seconds", 0),
            "error": item.get("error"),
            "ended_at": item.get("ended_at"),
        }

    # Determine overall status
    statuses = [s["status"] for s in services.values()]
    if not statuses:
        overall = "PENDING"
    elif all(s == "COMPLETED" for s in statuses):
        overall = "COMPLETED"
    elif any(s == "FAILED" for s in statuses):
        overall = "PARTIAL_FAILURE"
    else:
        overall = "RUNNING"

    return {
        "match_id": match_id,
        "overall_status": overall,
        "services": services,
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for submitting jobs"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Submit inference jobs to AWS Batch")
    parser.add_argument("--match-id", required=True, help="Match identifier")
    parser.add_argument("--stream-url", required=True, help="Stream URL")
    parser.add_argument("--config", required=True, help="Path to game config YAML")
    parser.add_argument("--enable-resume", action="store_true", help="Enable resume from last position")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually submit jobs")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for HLS metadata")

    args = parser.parse_args()

    # Load config
    from config.loader import ConfigLoader
    game_config = ConfigLoader.load_from_yaml(args.config)

    # Submit jobs
    result = submit_match_jobs(
        match_id=args.match_id,
        stream_url=args.stream_url,
        game_config=game_config.model_dump(),
        enable_resume=args.enable_resume,
        wait_for_hls_metadata=not args.no_wait,
        dry_run=args.dry_run,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
