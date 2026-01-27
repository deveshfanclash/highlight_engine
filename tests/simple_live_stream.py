#!/usr/bin/env python3
"""Creates a live HLS stream by looping a video and cleaning old segments."""

import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a live HLS stream from a video")
    parser.add_argument("-v", "--video", type=str, default="/Users/spectatr/Downloads/test_vlm.mp4",
                        help="Path to input video")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent / "hls_output",
                        help="Output directory for HLS files")
    parser.add_argument("-s", "--segment-time", type=int, default=2,
                        help="Segment duration in seconds (default: 2)")
    parser.add_argument("-m", "--max-segments", type=int, default=5,
                        help="Max segments to keep (default: 5)")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(exist_ok=True)
    # playlist = args.output_dir / "stream.m3u8"

    # FFmpeg command for live HLS with looping input
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-re",
        "-i", args.video,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-f", "hls",
        "-hls_time", str(args.segment_time),
        "-hls_list_size", str(args.max_segments),
        "-hls_flags", "delete_segments+indepdendent_segments",
        # str(playlist)
        str(args.output_dir / "segments_%v_%Y%m%dT%H%M%S_%%05d.ts")
    ]

    # print(f"Stream URL: {playlist}")
    print(f"Stream URL: {args.output_dir / 'segments_%v_%Y%m%dT%H%M%S_%%05d.ts'}")
    print("Press Ctrl+C to stop")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()

#Copy command to terminal
# python tests/create_live_stream.py -s 6 -m 10 -v /Users/spectatr/Downloads/test_vlm.mp4