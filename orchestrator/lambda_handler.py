"""
AWS Lambda Handler for Inference Job Submission

Entry point for triggering inference jobs from:
- API Gateway
- EventBridge
- Direct Lambda invocation

Environment Variables Required:
- MONGO_URI: MongoDB connection string
- DYNAMODB_TABLE: DynamoDB table name
- BATCH_GPU_QUEUE: AWS Batch GPU queue name
- BATCH_CPU_QUEUE: AWS Batch CPU queue name
"""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for submitting inference jobs.

    Event format:
    {
        "match_id": "match_123",
        "stream_url": "https://cdn.example.com/stream.m3u8",
        "game_id": "football",           # OR
        "config_path": "s3://bucket/config.yaml",  # OR
        "game_config": {...},             # Direct config
        "enable_resume": false,
        "wait_for_hls_metadata": true
    }

    Returns:
    {
        "statusCode": 200,
        "body": {
            "match_id": "match_123",
            "jobs": [...],
            "resume_position": {...}
        }
    }
    """
    try:
        # Parse event (handle API Gateway format)
        if "body" in event:
            # API Gateway event
            body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        else:
            body = event

        match_id = body.get("match_id")
        stream_url = body.get("stream_url")
        enable_resume = body.get("enable_resume", False)
        wait_for_hls = body.get("wait_for_hls_metadata", True)

        if not match_id or not stream_url:
            return _error_response(400, "match_id and stream_url are required")

        # Load game config from one of the sources
        game_config = _load_game_config(body)
        if game_config is None:
            return _error_response(400, "Must provide game_id, config_path, or game_config")

        # Submit jobs
        from orchestrator.batch_submitter import submit_match_jobs

        result = submit_match_jobs(
            match_id=match_id,
            stream_url=stream_url,
            game_config=game_config,
            enable_resume=enable_resume,
            wait_for_hls_metadata=wait_for_hls,
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result, default=str),
        }

    except Exception as e:
        logger.exception(f"Error processing request: {e}")
        return _error_response(500, str(e))


def status_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for checking match status.

    Event format:
    {
        "match_id": "match_123"
    }
    """
    try:
        if "body" in event:
            body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        else:
            body = event

        match_id = body.get("match_id")
        if not match_id:
            return _error_response(400, "match_id is required")

        from orchestrator.batch_submitter import check_match_status

        result = check_match_status(match_id)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result, default=str),
        }

    except Exception as e:
        logger.exception(f"Error checking status: {e}")
        return _error_response(500, str(e))


def _load_game_config(body: Dict[str, Any]) -> Dict[str, Any]:
    """Load game config from various sources"""

    # Option 1: Direct config in request
    if "game_config" in body:
        return body["game_config"]

    # Option 2: Load from MongoDB by game_id
    if "game_id" in body:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MONGO_URI env var required for game_id lookup")

        from config.loader import ConfigLoader
        loader = ConfigLoader(mongo_uri=mongo_uri)
        config = loader.load_game_config(body["game_id"])
        if config:
            return config.model_dump()
        raise ValueError(f"Game config not found for game_id: {body['game_id']}")

    # Option 3: Load from S3
    if "config_path" in body:
        config_path = body["config_path"]

        if config_path.startswith("s3://"):
            # Download from S3
            import boto3
            import tempfile

            s3 = boto3.client("s3")
            parts = config_path[5:].split("/", 1)
            bucket, key = parts[0], parts[1]

            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                s3.download_file(bucket, key, f.name)
                local_path = f.name

            from config.loader import ConfigLoader
            config = ConfigLoader.load_from_yaml(local_path)
            os.unlink(local_path)
            return config.model_dump()
        else:
            # Local path (for testing)
            from config.loader import ConfigLoader
            config = ConfigLoader.load_from_yaml(config_path)
            return config.model_dump()

    return None


def _error_response(status_code: int, message: str) -> Dict[str, Any]:
    """Create error response"""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }
