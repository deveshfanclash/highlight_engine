#!/usr/bin/env python3
"""
Creates a production-matching live HLS stream with multi-variant ABR.

Settings derived from:
- Production ffprobe output (480p + 1080p variants)
- video-segment-utility repo (GOP, profile, preset settings)
- generate_match_stream operation (GOP calculation)
"""

import argparse
import atexit
import shutil
import signal
import subprocess
import math
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a production-like live HLS stream")
    parser.add_argument("-v", "--video", type=str, default="/Users/spectatr/Downloads/test_vlm.mp4",
                        help="Path to input video")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent / "hls_output",
                        help="Output directory for HLS files")
    parser.add_argument("-s", "--segment-time", type=int, default=6,
                        help="Segment duration in seconds (default: 6)")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep files after exit (default: delete)")
    return parser.parse_args()


def get_video_fps(video_path: str) -> float:
    """Get FPS from video using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", video_path
        ]
        output = subprocess.check_output(cmd).decode().strip()
        num, denom = map(int, output.split("/"))
        return num / denom
    except Exception as e:
        print(f"Failed to get FPS, defaulting to 30: {e}")
        return 30.0


def has_audio(video_path: str) -> bool:
    """Check if video has an audio stream."""
    try:
        cmd = [
            "ffprobe", "-v", "0", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", video_path
        ]
        output = subprocess.check_output(cmd).decode().strip()
        return len(output) > 0
    except Exception:
        return False


def cleanup(output_dir: Path, keep_files: bool):
    """Clean up output directory on exit."""
    if keep_files:
        print(f"\nFiles kept at: {output_dir}")
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"\nCleaned up: {output_dir}")


def main():
    args = parse_args()
    args.output_dir.mkdir(exist_ok=True)
    master_playlist = args.output_dir / "segments.m3u8"

    # Register cleanup on exit
    atexit.register(cleanup, args.output_dir, args.keep_files)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # Get source FPS for GOP calculation
    fps = get_video_fps(args.video)
    gop = int(math.ceil(2.4 * fps))  # GOP = 2.4 * fps (from generate_match_stream)
    keyint_min = int(gop / 2)  # Half of GOP (similar to video-segment-utility ratio)

    # Check if video has audio
    audio_exists = has_audio(args.video)
    audio_input = "0" if audio_exists else "1"  # Use silent audio from input 1 if no audio

    print(f"Source FPS: {fps:.2f}, GOP: {gop}, Keyint min: {keyint_min}")
    print(f"Audio: {'from source' if audio_exists else 'generating silent'}")

    # Build input section
    inputs = [
        "-stream_loop", "-1",
        "-re",
        "-i", args.video,
    ]
    # Add silent audio source if video has no audio
    if not audio_exists:
        inputs.extend(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"])

    cmd = [
        "ffmpeg", "-y",
        *inputs,

        # 480p stream - H.264 High profile (~625 kbps)
        "-map", "0:v", "-map", f"{audio_input}:a",
        "-c:v:0", "libx264",
        "-profile:v:0", "high",
        "-level:v:0", "4.2",
        "-preset", "medium",
        "-crf:v:0", "23",
        "-b:v:0", "500k", "-maxrate:v:0", "750k", "-bufsize:v:0", "1000k",
        "-g:v:0", str(gop),
        "-keyint_min:v:0", str(keyint_min),
        "-sc_threshold:v:0", "0",
        "-s:v:0", "854x480",
        "-pix_fmt:v:0", "yuv420p",
        "-c:a:0", "aac", "-b:a:0", "192k", "-ar:a:0", "48000", "-ac:a:0", "2",

        # 1080p stream - H.264 High profile (~11 Mbps)
        "-map", "0:v", "-map", f"{audio_input}:a",
        "-c:v:1", "libx264",
        "-profile:v:1", "high",
        "-level:v:1", "4.2",
        "-preset", "medium",
        "-crf:v:1", "23",
        "-b:v:1", "11000k", "-maxrate:v:1", "16500k", "-bufsize:v:1", "22000k",
        "-g:v:1", str(gop),
        "-keyint_min:v:1", str(keyint_min),
        "-sc_threshold:v:1", "0",
        "-s:v:1", "1920x1080",
        "-pix_fmt:v:1", "yuv420p",
        "-c:a:1", "aac", "-b:a:1", "192k", "-ar:a:1", "48000", "-ac:a:1", "2",

        # HLS output settings (production-like: keep all segments)
        "-f", "hls",
        "-hls_time", str(args.segment_time),
        "-hls_list_size", "0",  # Keep all segments in playlist (like production)
        "-hls_flags", "independent_segments",  # No delete_segments (production-like)
        "-hls_segment_filename", str(args.output_dir / "segments_%v_%Y%m%dT%H%M%S_%%05d.ts"),
        "-master_pl_name", "segments.m3u8",
        "-var_stream_map", "v:0,a:0,name:480p v:1,a:1,name:1080p",
        str(args.output_dir / "segments_%v.m3u8")
    ]

    print(f"Master playlist: {master_playlist}")
    print(f"Variants: segments_480p.m3u8, segments_1080p.m3u8")
    print(f"Cleanup on exit: {not args.keep_files}")
    print("Press Ctrl+C to stop")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass  # Cleanup handled by atexit


if __name__ == "__main__":
    main()

# Usage:
# python tests/hls_video_processing.py -v /Users/spectatr/Downloads/test_vlm.mp4
#
# Keep files after exit:
# python tests/hls_video_processing.py --keep-files
#
# Use RAM disk for zero disk usage (macOS):
# diskutil erasevolume HFS+ 'HLS' `hdiutil attach -nomount ram://2097152`
# python tests/hls_video_processing.py -o /Volumes/HLS
# diskutil eject /Volumes/HLS
