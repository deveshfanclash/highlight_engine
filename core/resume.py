"""
Resume Logic

Handles resume position calculation for services based on different modes:
- start: Begin from frame 0 (fresh start)
- current: Resume from last frame written by this service
- latest: Start from current stream position
"""

import logging
from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy import for boto3 to avoid import errors when not installed
boto3 = None
ClientError = Exception  # Fallback exception type


class ResumeMode(str, Enum):
    """Resume modes for starting services"""
    START = "start"      # Fresh start from frame 0
    CURRENT = "current"  # Resume from last written frame
    LATEST = "latest"    # Start from current stream position


@dataclass
class ResumePosition:
    """Position information for resuming a service"""
    frame_number: int
    segment_number: int
    timestamp_ms: int = 0


def get_resume_position(
    mode: ResumeMode,
    match_id: str,
    service_id: str,
    source_url: str,
    table_name: str = "inference_results",
    region: str = "us-east-1",
) -> ResumePosition:
    """
    Get the resume position based on the specified mode.

    Args:
        mode: Resume mode (start, current, latest)
        match_id: Match identifier
        service_id: Service identifier
        source_url: Stream/file URL (used for 'latest' mode)
        table_name: DynamoDB table name for querying last written frame
        region: AWS region

    Returns:
        ResumePosition with frame_number, segment_number, and timestamp_ms
    """
    if mode == ResumeMode.START:
        return _get_start_position()
    elif mode == ResumeMode.CURRENT:
        return _get_current_position(match_id, service_id, table_name, region)
    elif mode == ResumeMode.LATEST:
        return _get_latest_position(source_url)
    else:
        logger.warning(f"Unknown resume mode: {mode}, defaulting to start")
        return _get_start_position()


def _get_start_position() -> ResumePosition:
    """
    Get position for fresh start.

    Returns:
        ResumePosition starting from frame 0
    """
    logger.info("Resume mode: START - beginning from frame 0")
    return ResumePosition(
        frame_number=0,
        segment_number=0,
        timestamp_ms=0
    )


def _get_current_position(
    match_id: str,
    service_id: str,
    table_name: str,
    region: str
) -> ResumePosition:
    """
    Get position from last written frame in DB.

    Queries the inference_results table for the highest frame number
    written by this match_id + service_id combination.

    Args:
        match_id: Match identifier
        service_id: Service identifier
        table_name: DynamoDB table name
        region: AWS region

    Returns:
        ResumePosition with last written frame + 1, or start position if none found
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.warning("boto3 not installed, cannot query DB for resume position. Starting from frame 0.")
        return _get_start_position()

    try:
        dynamodb = boto3.resource("dynamodb", region_name=region)
        table = dynamodb.Table(table_name)

        pk = f"{match_id}#{service_id}"

        # Query for the highest SK (frame_number) for this PK
        response = table.query(
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": pk},
            ScanIndexForward=False,  # Descending order
            Limit=1,
            ProjectionExpression="sk, timestamp_ms"
        )

        if response.get("Items"):
            last_item = response["Items"][0]
            last_frame = int(last_item["sk"])
            last_timestamp = int(last_item.get("timestamp_ms", 0))

            # Resume from the next frame
            resume_frame = last_frame + 1

            logger.info(
                f"Resume mode: CURRENT - last frame was {last_frame}, "
                f"resuming from frame {resume_frame}"
            )

            return ResumePosition(
                frame_number=resume_frame,
                segment_number=0,  # Will be calculated by input handler
                timestamp_ms=last_timestamp
            )
        else:
            logger.info(
                f"Resume mode: CURRENT - no existing data found for {pk}, "
                f"starting from frame 0"
            )
            return _get_start_position()

    except ClientError as e:
        logger.warning(
            f"Failed to query DynamoDB for resume position: {e}. "
            f"Falling back to start position."
        )
        return _get_start_position()
    except Exception as e:
        logger.warning(
            f"Unexpected error getting resume position: {e}. "
            f"Falling back to start position."
        )
        return _get_start_position()


def _get_latest_position(source_url: str) -> ResumePosition:
    """
    Get position for starting from current stream position.

    For live streams, this would query the stream for the current position.
    For files, this typically means starting from the beginning.

    Note: Full implementation would require stream-specific logic
    (e.g., parsing HLS manifest for live edge).

    Args:
        source_url: Stream or file URL

    Returns:
        ResumePosition for current stream position
    """
    # For now, return a position that signals "start from live edge"
    # The input handler will interpret segment_number=-1 as "start from latest"
    logger.info(f"Resume mode: LATEST - starting from current stream position")

    # Check if it's likely a file (not a live stream)
    is_file = (
        source_url.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        or source_url.startswith('/')
        or source_url.startswith('file://')
    )

    if is_file:
        logger.info("Source appears to be a file, starting from beginning")
        return _get_start_position()

    # For live streams, use special marker
    # segment_number=-1 tells the input handler to start from live edge
    return ResumePosition(
        frame_number=-1,  # Signal to start from live
        segment_number=-1,
        timestamp_ms=-1
    )
