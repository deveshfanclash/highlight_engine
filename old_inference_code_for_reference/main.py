import json
import os
import multiprocessing
import time
from pathlib import Path
from dotenv import load_dotenv
from utils.log_utils import logger
from src.live_stream_processing.real_time_inference import real_time_inference
from src.live_stream_processing.camera_view_detection import camera_view_stream
from src.live_stream_processing.hls_stream_metadata import hls_stream_metadata_write

load_dotenv(override=False)


def test(data):
    # message = {
    #     "matchId": "6877a6262456475972ba8fec",
    #     # "streamUrl": "https://highlights-cdn.spectatr.gg/hls/production/68ad7212e02bcdd9011ee6ab/67f8b5ea6d0b02e4b1742849/segments.m3u8",
    #     "streamUrl": "https://highlights-cdn.spectatr.gg/hls/staging/6877a6262456475972ba8fec/67876a3d86c44d602c6d57c1/segments.m3u8",
    #     "streamStatus": "started",
    #     "league": "league",
    #     "lang": "en",
    #     "game": {
    #         "id": "68695c52a425fc77ac37b8da",
    #         "name": "Volleyball",
    #     },
    #     "tournament": {
    #         "id": "68a6ef0f116ca0108d840934",
    #         "name": "Northern Super League 2025",
    #     }
    # }

    if data.get("streamStatus", None) == "started":  # Match stream started
        logger.info(f"Starting inference for match: {data.get('matchId', 'unknown')}")

        # Create process objects
        process1 = multiprocessing.Process(target=real_time_inference, args=(data,))
        process2 = multiprocessing.Process(target=camera_view_stream, args=(data,))
        process3 = multiprocessing.Process(
            target=hls_stream_metadata_write, args=(data,)
        )

        # Start hls_stream_metadata_write immediately
        logger.info("Starting hls_stream_metadata_write process immediately")
        process3.start()

        # Wait 30 seconds before starting the other two processes
        logger.info(
            "Waiting 30 seconds before starting real_time_inference and camera_view_stream processes"
        )
        time.sleep(30)

        # Start real_time_inference and camera_view_stream after 30 seconds
        logger.info("Starting real_time_inference and camera_view_stream processes")
        process1.start()
        process2.start()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Real-time inference")
    parser.add_argument(
        "message_path",
        help="Absolute or relative path to the JSON file (e.g. /tmp/event.json)",
    )
    args = parser.parse_args()
    return Path(args.message_path).expanduser().resolve()


def load_message(path):
    if not path.exists():
        raise SystemExit(f"Error: File not found: {path}")
    if not path.is_file():
        raise SystemExit(f"Error: {path} is not a file")

    with path.open("r", encoding="utf-8") as infile:
        try:
            return json.load(infile)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Error parsing JSON: {exc}")


if __name__ == "__main__":
    # Build data dict from environment variables
    data = {
        "matchId": os.getenv("MATCH_ID"),
        "streamUrl": os.getenv("STREAM_URL"),
        "streamStatus": os.getenv("STREAM_STATUS", "started"),
        "league": os.getenv("LEAGUE"),
        "lang": os.getenv("LANG"),
        "game": {
            "id": os.getenv("GAME_ID"),
            "name": os.getenv("GAME_NAME"),
        },
        "tournament": {
            "id": os.getenv("TOURNAMENT_ID"),
            "name": os.getenv("TOURNAMENT_NAME"),
        },
    }
    test(data)
