import os
import sys
import boto3
import requests
import re
import time
from decimal import Decimal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config
import pandas as pd
from boto3.dynamodb.conditions import Key
from pathlib import Path
sys.path.append(os.getcwd())
from config.settings import *

from utils import logger
config = Config(max_pool_connections=200)
session = boto3.session.Session()
dynamodb = session.resource(
    'dynamodb',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_DB"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY_DB"),
    region_name=env_cfg.AWS_REGION,
    config=config
)
    
def get_highest_resolution_variant(master_m3u8_url: str, timeout: int = 10) -> str:
    """
    From a master playlist, fetch the highest resolution variant playlist URL.
    """
    resp = requests.get(master_m3u8_url, timeout=timeout)
    resp.raise_for_status()
    base_url = master_m3u8_url.rsplit("/", 1)[0]

    lines = resp.text.splitlines()
    variants = []

    for i, line in enumerate(lines):
        if line.startswith("#EXT-X-STREAM-INF"):
            # Extract resolution if available
            res_match = re.search(r"RESOLUTION=(\d+)x(\d+)", line)
            if res_match and i+1 < len(lines):
                width, height = map(int, res_match.groups())
                playlist_url = lines[i+1].strip()
                if not playlist_url.startswith("http"):
                    playlist_url = f"{base_url}/{playlist_url}"
                variants.append((width*height, playlist_url))

    if not variants:
        raise ValueError("No variant streams found in master playlist")

    # Pick highest resolution by pixel count
    return max(variants, key=lambda x: x[0])[1]


def is_live(m3u8_url: str, timeout: int = 10) -> bool:
    """
    Check if a given HLS media playlist is LIVE.
    If PLAYLIST-TYPE not found, falls back to master playlist to get highest res variant.
    """
    try:
        resp = requests.get(m3u8_url, timeout=timeout)
        resp.raise_for_status()
        lines = [line.strip() for line in resp.text.splitlines() if line.strip()]

        if "#EXT-X-PLAYLIST-TYPE:VOD" in lines or "#EXT-X-ENDLIST" in lines:
            return False
        elif "#EXT-X-PLAYLIST-TYPE:EVENT" in lines:
            return True
        else:
            # If type is missing, maybe we were given master instead of media
            if any(line.startswith("#EXT-X-STREAM-INF") for line in lines):
                high_res_url = get_highest_resolution_variant(m3u8_url)
                return is_live(high_res_url)  # recurse into the media playlist
            else:
                # If it's a plain media playlist without ENDLIST → assume live
                return "#EXT-X-ENDLIST" not in lines

    except Exception as e:
        print(f"[ERROR] Failed to fetch {m3u8_url}: {e}")
        return False

def get_video_frame_metadata(match_id, stream_url, fps=60, segment_length=7):
    table = dynamodb.Table(env_cfg.TABLE_NAME_VIDEO_FRAMES_METADATA)
    # fps = 60 # NOTE: should make dynamic for get fps
    # segment_length = 2 # NOTE: should make dynamic for get average length of each segment
    N = fps * segment_length * 2   # e.g. 720

    try:
        logger.info(f"Fetching video frame metadata for match_id={match_id} with N={N}")
        if is_live(stream_url):
            resp = table.query(
                KeyConditionExpression=Key("match_id").eq(match_id),
                ScanIndexForward=False,   # latest first
                Limit=N
            )
            items = resp.get("Items", [])
            if not items:
                logger.warning(f"No partition key found for match_id={match_id}, returning defaults.")
                return {"segment_number": 0, "frame_number": 1}

            # Reverse to ascending order for frame processing if needed
            items = list(reversed(items))

            # Find highest segment_number
            max_seg = max(int(i["segment_number"]) for i in items)

            # Get first frame of that segment
            result = min(
                [i for i in items if int(i["segment_number"]) == max_seg],
                key=lambda x: int(x["frame_number"])
            )

            final_result = {
                "segment_number": int(result.get("segment_number", 0)),
                "frame_number": int(result.get("frame_number", 0)),
            }

            logger.info(f"Found latest metadata for match_id={match_id}: {final_result}")
        
            return final_result
        else:
            logger.info(f"Stream already completed for match_id={match_id}, Resume from segment 0")
            return {"segment_number": 0, "frame_number": 1}

    except Exception as e:
        logger.error(f"Error while fetching video frame metadata for match_id={match_id}: {str(e)}")
        return {"segment_number": 0, "frame_number": 1}

    finally:
        logger.debug(f"Completed metadata fetch process for match_id={match_id}")

def get_video_frame_metadata_v2(match_id, stream_url, fps=25, segment_length=7):
    table = dynamodb.Table(env_cfg.TABLE_NAME_VIDEO_FRAMES_METADATA)
    N = fps * segment_length * 2   # e.g. 720

    try:
        logger.info(f"Fetching video frame metadata for match_id={match_id} with N={N}")
        if is_live(stream_url):
            resp = table.query(
                KeyConditionExpression=Key("match_id").eq(match_id),
                ScanIndexForward=False,   # latest first
                Limit=N
            )
            items = resp.get("Items", [])
            if not items:
                logger.warning(f"No partition key found for match_id={match_id}, returning defaults.")
                return {"segment_number": 1, "frame_number": 1}

            # Reverse to ascending order for processing
            items = list(reversed(items))

            # Find all unique segment_numbers
            unique_segments = sorted({int(i["segment_number"]) for i in items})
            print("unique_segments: ", unique_segments)

            if len(unique_segments) < 2:
                logger.info(f"Only one segment found for match_id={match_id}, using segment {unique_segments[-1]}")
                target_seg = unique_segments[-1]
            else:
                # Get the second highest segment
                target_seg = unique_segments[-2]

            # Get item with max frame_number in that segment
            result = max(
                [i for i in items if int(i["segment_number"]) == target_seg],
                key=lambda x: int(x["frame_number"])
            )

            final_result = {
                "segment_number": int(result.get("segment_number")+1),
                "frame_number": int(result.get("frame_number"))+1,
            }

            logger.info(f"Found resume metadata for match_id={match_id}: {final_result}")
            return final_result
        else:
            logger.info(f"Stream already completed for match_id={match_id}, Resume from segment 0")
            return {"segment_number": 1, "frame_number": 1}
    except Exception as e:
        logger.error(f"Error while fetching video frame metadata for match_id={match_id}: {str(e)}")
        return {"segment_number": 1, "frame_number": 1}
    finally:
        logger.debug(f"Completed metadata fetch process for match_id={match_id}")

def data_main():
    pass
    match_id = "67f8a86a836c9a57ecb6a016"
    stream_url="https://highlights-cdn.spectatr.gg/hls/production/67f8a86a836c9a57ecb6a016/67f8a71e6d0b02e4b1742839/segments.m3u8"
    get_video_frame_metadata(match_id, stream_url)
    get_video_frame_metadata_v2(match_id, stream_url)

if __name__=="__main__":
    data_main()