"""
HLS Stream Metadata Service

Extracts frame-level metadata from HLS streams:
- Polls m3u8 playlist for new .ts segments
- Uses ffprobe to extract PTS times
- Writes metadata to DynamoDB for resume functionality

This service must run BEFORE other services (OD, Camera View) to enable resume.

Extracted from old_inference_code_for_reference/src/live_stream_processing/hls_stream_metadata.py
"""

import os
import json
import time
import signal
import logging
import subprocess
from dataclasses import dataclass
from typing import Iterator, Optional, Set, Dict, Any, List
from decimal import Decimal
from urllib.parse import urljoin
from datetime import datetime

import requests

from db.dynamo import DynamoDBWriter, DynamoDBWriterConfig

logger = logging.getLogger(__name__)


@dataclass
class HLSMetadataServiceConfig:
    """Configuration for HLS Metadata Service"""
    match_id: str
    stream_url: str

    # Polling settings
    poll_interval: int = 5  # Seconds between m3u8 polls
    timeout_no_segment: int = 60  # Stop if no new segment for this long
    resolution_preference: str = "_480p.m3u8"  # Use low res for metadata (faster)

    # Database settings
    db_table_name: str = "video_frames_metadata"
    batch_size: int = 500

    # Local testing
    local_output_dir: Optional[str] = None  # If set, write to files instead of DB


class HLSMetadataService:
    """
    Extracts and stores frame-level metadata from HLS streams.

    This service:
    1. Polls the m3u8 playlist continuously for new .ts segments
    2. Uses ffprobe to extract PTS times from each segment
    3. Writes metadata to DynamoDB (or local files for testing)

    Other services query this metadata to resume from the correct position.

    Usage:
        config = HLSMetadataServiceConfig(
            match_id="match_123",
            stream_url="https://example.com/stream.m3u8"
        )
        service = HLSMetadataService(config)
        service.run()
    """

    def __init__(self, config: HLSMetadataServiceConfig):
        self.config = config
        self._running = False
        self._db_writer: Optional[DynamoDBWriter] = None
        self._frame_number = 1
        self._seen_segments: Set[str] = set()

    def _get_best_stream_url(self) -> str:
        """Get low-resolution variant for metadata extraction"""
        from core.utils import get_best_stream_url
        return get_best_stream_url(
            self.config.stream_url,
            self.config.resolution_preference
        )

    def _stream_ts_segments(self, m3u8_url: str) -> Iterator[str]:
        """
        Yield new .ts segment URLs as they appear in the playlist.

        Continuously polls the m3u8 playlist and yields only NEW segments.
        Stops when #EXT-X-ENDLIST is found or timeout occurs.
        """
        base_url = m3u8_url.rsplit("/", 1)[0] + "/"
        last_segment_time = time.time()

        while self._running:
            try:
                resp = requests.get(m3u8_url, timeout=10)
                resp.raise_for_status()
                lines = resp.text.splitlines()

                # Check for end of stream
                if "#EXT-X-ENDLIST" in lines:
                    logger.info("Found #EXT-X-ENDLIST, stream complete")
                    # Yield any remaining segments
                    for line in lines:
                        if line.endswith(".ts"):
                            full_url = urljoin(base_url, line)
                            if full_url not in self._seen_segments:
                                self._seen_segments.add(full_url)
                                yield full_url
                    break

                # Extract new segments
                for line in lines:
                    if line.endswith(".ts"):
                        full_url = urljoin(base_url, line)
                        if full_url not in self._seen_segments:
                            self._seen_segments.add(full_url)
                            last_segment_time = time.time()
                            yield full_url

                # Check timeout
                if time.time() - last_segment_time > self.config.timeout_no_segment:
                    logger.warning(
                        f"No new segments for {self.config.timeout_no_segment}s, stopping"
                    )
                    break

                # Wait before next poll
                time.sleep(self.config.poll_interval)

            except Exception as e:
                logger.error(f"Error polling m3u8: {e}")
                time.sleep(self.config.poll_interval)

    def _process_segment(self, ts_url: str) -> int:
        """
        Extract frame metadata from a .ts segment using ffprobe.

        Returns:
            Number of frames processed
        """
        segment_name = os.path.basename(ts_url)

        # Extract segment number from filename (e.g., "segment_123.ts" -> 123)
        try:
            segment_number = int(segment_name.rsplit("_", 1)[-1].split(".")[0])
        except (ValueError, IndexError):
            segment_number = len(self._seen_segments)

        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_frames",  # Required to actually read frame-by-frame data
            "-show_entries", "frame=pkt_pts_time,pkt_dts_time,best_effort_timestamp_time",
            "-of", "json",
            ts_url
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )

            if proc.returncode != 0:
                logger.warning(f"ffprobe failed for {ts_url}: {proc.stderr.strip()}")
                return 0

            data = json.loads(proc.stdout)
            frames = data.get("frames", [])

            # Debug: log if no frames found
            if not frames:
                logger.warning(f"No frames found in ffprobe output for {segment_name}. "
                              f"Raw output length: {len(proc.stdout)} bytes")

            batch_items = []
            frames_processed = 0

            for idx, frame_data in enumerate(frames):
                # Try pkt_pts_time first, fallback to pkt_dts_time or best_effort_timestamp_time
                pts_time_str = frame_data.get("pkt_pts_time") or \
                               frame_data.get("pkt_dts_time") or \
                               frame_data.get("best_effort_timestamp_time")

                if pts_time_str is None:
                    continue

                pts_time = Decimal(pts_time_str)

                item = {
                    "pk": self.config.match_id,  # Partition key
                    "sk": float(pts_time),  # Sort key (ptstime)
                    "match_id": self.config.match_id,
                    "segment": segment_name,
                    "ptstime": float(pts_time),
                    "ts_frame": idx,
                    "frame_number": self._frame_number,
                    "segment_number": segment_number,
                }

                batch_items.append(item)
                self._frame_number += 1
                frames_processed += 1

                # Batch write
                if len(batch_items) >= self.config.batch_size:
                    self._write_batch(batch_items)
                    batch_items = []

            # Final batch
            if batch_items:
                self._write_batch(batch_items)

            logger.info(
                f"Processed segment {segment_name}: {frames_processed} frames, "
                f"total frame_number={self._frame_number}"
            )

            return frames_processed

        except subprocess.TimeoutExpired:
            logger.error(f"ffprobe timed out for {ts_url}")
            return 0
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ffprobe output: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error processing segment {ts_url}: {e}")
            return 0

    def _write_batch(self, items: List[Dict[str, Any]]):
        """Write batch of items to DB or local file"""
        if self.config.local_output_dir:
            self._write_local(items)
        elif self._db_writer:
            for item in items:
                self._db_writer.write_item(item)

    def _write_local(self, items: List[Dict[str, Any]]):
        """Write to local JSON file for testing"""
        os.makedirs(self.config.local_output_dir, exist_ok=True)
        output_file = os.path.join(
            self.config.local_output_dir,
            f"{self.config.match_id}_hls_metadata.jsonl"
        )

        with open(output_file, "a") as f:
            for item in items:
                f.write(json.dumps(item, default=str) + "\n")

    def run(self):
        """Main service loop"""
        self._running = True
        self._setup_signal_handlers()

        logger.info(f"Starting HLS Metadata Service for match {self.config.match_id}")

        # Initialize DB writer if not using local output
        if not self.config.local_output_dir:
            db_config = DynamoDBWriterConfig.from_env(self.config.db_table_name)
            db_config.batch_size = self.config.batch_size
            self._db_writer = DynamoDBWriter(db_config)
            self._db_writer.start_background_writer()

        try:
            m3u8_url = self._get_best_stream_url()
            logger.info(f"Using stream URL: {m3u8_url}")

            for ts_url in self._stream_ts_segments(m3u8_url):
                if not self._running:
                    break
                self._process_segment(ts_url)
                time.sleep(0.1)  # Small delay to avoid overloading

            logger.info(
                f"HLS Metadata Service finished. "
                f"Total frames: {self._frame_number - 1}"
            )

        except Exception as e:
            logger.error(f"Error in HLS Metadata Service: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the service"""
        logger.info("Stopping HLS Metadata Service")
        self._running = False

        if self._db_writer:
            self._db_writer.stop()

    def _setup_signal_handlers(self):
        """Set up graceful shutdown handlers"""
        def handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self.stop()

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)


# =============================================================================
# RESUME HELPER FUNCTIONS
# =============================================================================

def get_resume_position(
    match_id: str,
    stream_url: str,
    fps: float = 25.0,
    segment_length: int = 7,
    db_table_name: str = "video_frames_metadata"
) -> Dict[str, int]:
    """
    Get the position to resume from based on stored metadata.

    Queries DynamoDB for the latest frame metadata and returns
    the segment and frame number to start from.

    Args:
        match_id: Match identifier
        stream_url: Stream URL (to check if still live)
        fps: Frames per second
        segment_length: Average segment length in seconds
        db_table_name: DynamoDB table name

    Returns:
        Dict with "segment_number" and "frame_number"
    """
    try:
        import boto3
        from boto3.dynamodb.conditions import Key
    except ImportError:
        logger.warning("boto3 not available, returning defaults")
        return {"segment_number": 1, "frame_number": 1}

    # Check if stream is still live
    if not _is_stream_live(stream_url):
        logger.info("Stream completed, starting from beginning")
        return {"segment_number": 1, "frame_number": 1}

    try:
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_DB"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY_DB"),
        )
        table = dynamodb.Table(db_table_name)

        # Query recent items
        N = int(fps * segment_length * 2)
        resp = table.query(
            KeyConditionExpression=Key("match_id").eq(match_id),
            ScanIndexForward=False,  # Latest first
            Limit=N
        )

        items = resp.get("Items", [])
        if not items:
            logger.info(f"No metadata found for {match_id}, starting from beginning")
            return {"segment_number": 1, "frame_number": 1}

        # Find unique segments
        unique_segments = sorted({int(i["segment_number"]) for i in items})

        if len(unique_segments) < 2:
            target_seg = unique_segments[-1]
        else:
            # Use second-highest segment (avoid incomplete segment)
            target_seg = unique_segments[-2]

        # Get max frame in target segment
        segment_items = [i for i in items if int(i["segment_number"]) == target_seg]
        result = max(segment_items, key=lambda x: int(x["frame_number"]))

        resume_position = {
            "segment_number": int(result["segment_number"]) + 1,
            "frame_number": int(result["frame_number"]) + 1,
        }

        logger.info(f"Resume position for {match_id}: {resume_position}")
        return resume_position

    except Exception as e:
        logger.error(f"Error getting resume position: {e}")
        return {"segment_number": 1, "frame_number": 1}


def _is_stream_live(m3u8_url: str, timeout: int = 10) -> bool:
    """Check if HLS stream is still live (not ended)"""
    try:
        resp = requests.get(m3u8_url, timeout=timeout)
        resp.raise_for_status()
        content = resp.text

        if "#EXT-X-ENDLIST" in content:
            return False
        if "#EXT-X-PLAYLIST-TYPE:VOD" in content:
            return False

        return True

    except Exception as e:
        logger.warning(f"Error checking stream status: {e}")
        return False


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

    parser = argparse.ArgumentParser(description="HLS Stream Metadata Service")
    parser.add_argument("--match-id", required=True, help="Match identifier")
    parser.add_argument("--stream-url", required=True, help="Stream URL")
    parser.add_argument("--poll-interval", type=int, default=5, help="Poll interval (seconds)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for no new segments")
    parser.add_argument("--db-table", default="video_frames_metadata", help="DynamoDB table")
    parser.add_argument("--local-output", help="Local output directory (for testing)")

    args = parser.parse_args()

    config = HLSMetadataServiceConfig(
        match_id=args.match_id,
        stream_url=args.stream_url,
        poll_interval=args.poll_interval,
        timeout_no_segment=args.timeout,
        db_table_name=args.db_table,
        local_output_dir=args.local_output,
    )

    service = HLSMetadataService(config)
    service.run()


if __name__ == "__main__":
    main()
