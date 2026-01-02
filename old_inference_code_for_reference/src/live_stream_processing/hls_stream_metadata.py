import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.settings import *
import subprocess
import json
import time
import logging
from logging.handlers import RotatingFileHandler
import requests
from decimal import Decimal
from urllib.parse import urljoin
from database.dynamodb_writer import write_frames_metadata_to_dynamo
from utils import get_best_stream_url
from dotenv import load_dotenv

# Load env
load_dotenv()
log_file = env_cfg.LOG_FILE_BASE_PATH/ "hls_stream_metadata.log" 
# Configure logging with rotation (max 50MB per file, keep last 3 backups)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=3),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)
logger.info("Logger initialized successfully.")

def stream_ts_segments(m3u8_url: str, poll_interval: int = 5, timeout_no_segment: int = 60):
    """
    Continuously fetch unique TS segments from a growing m3u8 playlist until #EXT-X-ENDLIST is found.
    Stops if no new TS segment arrives within `timeout_no_segment` seconds.
    """
    seen_segments = set()
    endlist_found = False
    base_url = m3u8_url.rsplit("/", 1)[0] + "/"

    last_segment_time = time.time()

    while not endlist_found:
        try:
            logger.info(f"Fetching m3u8 playlist: {m3u8_url}")
            resp = requests.get(m3u8_url, timeout=10)
            resp.raise_for_status()
            lines = resp.text.splitlines()

            # Check if VOD finished
            if "#EXT-X-ENDLIST" in lines:
                endlist_found = True
                logger.info("Found #EXT-X-ENDLIST, finishing stream processing.")

            # Extract unique TS URLs
            for line in lines:
                if line.endswith(".ts"):
                    full_url = urljoin(base_url, line)
                    if full_url not in seen_segments:
                        seen_segments.add(full_url)
                        last_segment_time = time.time()  # reset timeout
                        yield full_url   # <-- only yield new unique segment

            # Timeout if nothing new
            if time.time() - last_segment_time > timeout_no_segment:
                logger.error(
                    f"No new TS segments for {timeout_no_segment} seconds. Stopping."
                )
                raise TimeoutError(
                    f"No new TS segments for {timeout_no_segment} seconds"
                )

        except Exception as e:
            logger.error(f"Failed to fetch TS segments: {e}", exc_info=True)
            raise

        if not endlist_found:
            time.sleep(poll_interval)

def get_ts_segments(m3u8_url: str):
    """Fetch the m3u8 file and extract .ts segment URLs."""
    try:
        logger.info(f"Fetching m3u8 playlist: {m3u8_url}")
        resp = requests.get(m3u8_url, timeout=10)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        base_url = m3u8_url.rsplit("/", 1)[0] + "/"
        segments = [urljoin(base_url, line) for line in lines if line.endswith(".ts")]
        logger.info(f"Found {len(segments)} segments in playlist")
        return segments
    except Exception as e:
        logger.error(f"Failed to fetch TS segments from {m3u8_url}: {e}", exc_info=True)
        return []

def process_ts_segment(match_id: str, ts_url: str, frame_number: int, batch_size=500):
    """Run ffprobe on a .ts segment and upload frame metadata."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pkt_pts_time",
        "-of", "json",
        ts_url
    ]
    batch_items = []
    segment_name = os.path.basename(ts_url)

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
            return frame_number

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ffprobe output for {ts_url}: {e}")
            return frame_number

        frames = data.get("frames", [])
        for idx, f in enumerate(frames):
            if "pkt_pts_time" not in f:
                continue
            pts_time = Decimal(f.get("pkt_pts_time", "0"))
            entry = {
                "match_id": match_id,
                "segment": segment_name,
                "ptstime": pts_time,
                "ts_frame": idx,
                "frame_number": frame_number,
                "segment_number": int(segment_name.rsplit("_", 1)[-1].split(".")[0])
            }
            batch_items.append(entry)
            frame_number += 1

            if len(batch_items) >= batch_size:
                try:
                    write_frames_metadata_to_dynamo(batch_items)
                    logger.info(f"Uploaded {len(batch_items)} frames (last={frame_number}) from {segment_name}")
                except Exception as e:
                    logger.error(f"Failed to upload batch to DynamoDB: {e}", exc_info=True)
                batch_items = []

        # Final flush
        if batch_items:
            try:
                write_frames_metadata_to_dynamo(batch_items)
                logger.info(f"Uploaded final {len(batch_items)} frames for {segment_name}")
            except Exception as e:
                logger.error(f"Failed to upload final batch to DynamoDB: {e}", exc_info=True)

    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timed out for {ts_url}")
    except Exception as e:
        logger.error(f"Unexpected error processing segment {ts_url}: {e}", exc_info=True)
    finally:
        time.sleep(0.1)  # Always sleep a bit to avoid overloading Dynamo
    return frame_number

def hls_stream_metadata_write(message, batch_size=500):
    match_id = message.get("matchId")
    m3u8_url = message.get("streamUrl")
    m3u8_url = get_best_stream_url(m3u8_url, "_480p.m3u8") # Should use as low resolution we hvae as we just need to extract metadata
    
    try:
        frame_number = 1
        for ts_url in stream_ts_segments(m3u8_url, poll_interval=5):
            frame_number = process_ts_segment(match_id, ts_url, frame_number, batch_size=batch_size)

        logger.info(f"Finished processing all segments. Last frame={frame_number}")
    except Exception as e:
        logger.error(f"Error in stream_all_frames: {e}", exc_info=True)

def main():
    message = {
        "matchId": "689c7bf44c7337560952fb94",
        "streamUrl": "https://highlights-cdn.spectatr.gg/hls/staging/689c7bf44c7337560952fb94/67876a3d86c44d602c6d57c1/segments.m3u8",
        "streamStatus": "started",
    }
    try:
        hls_stream_metadata_write(message)
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
    finally:
        logger.info("Shutting down gracefully.")

if __name__ == "__main__":
    main()
