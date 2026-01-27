#!/usr/bin/env python3
"""Creates a live HLS stream matching production (multi-variant ABR)."""

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
    parser.add_argument("-s", "--segment-time", type=int, default=6,
                        help="Segment duration in seconds (default: 6)")
    parser.add_argument("-m", "--max-segments", type=int, default=10,
                        help="Max segments to keep (default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(exist_ok=True)
    master_playlist = args.output_dir / "segments.m3u8"

    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-re",
        "-i", args.video,

        # 480p stream - H.264 High profile (~625 kbps)
        "-map", "0:v", "-map", "0:a",
        "-c:v:0", "libx264", "-profile:v:0", "high", "-preset", "fast",
        "-b:v:0", "500k", "-maxrate:v:0", "600k", "-bufsize:v:0", "1200k",
        "-s:v:0", "854x480",
        "-c:a:0", "aac", "-b:a:0", "128k", "-ar:a:0", "48000", "-ac:a:0", "2",

        # 1080p stream - H.264 Main profile (~11 Mbps)
        "-map", "0:v", "-map", "0:a",
        "-c:v:1", "libx264", "-profile:v:1", "main", "-preset", "fast",
        "-b:v:1", "11000k", "-maxrate:v:1", "11500k", "-bufsize:v:1", "23000k",
        "-s:v:1", "1920x1080",
        "-c:a:1", "aac", "-b:a:1", "128k", "-ar:a:1", "48000", "-ac:a:1", "2",

        # HLS output settings
        "-f", "hls",
        "-hls_time", str(args.segment_time),
        "-hls_list_size", str(args.max_segments),
        "-hls_flags", "delete_segments+independent_segments",
        "-hls_segment_filename", str(args.output_dir / "segments_%v_%Y%m%dT%H%M%S_%%05d.ts"),
        "-master_pl_name", "segments.m3u8",
        "-var_stream_map", "v:0,a:0,name:480p v:1,a:1,name:1080p",
        str(args.output_dir / "segments_%v.m3u8")
    ]

    print(f"Master playlist: {master_playlist}")
    print(f"Variants: segments_480p.m3u8, segments_1080p.m3u8")
    print("Press Ctrl+C to stop")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()

# python tests/create_live_stream.py -s 6 -m 10 -v /Users/spectatr/Downloads/test_vlm.mp4
