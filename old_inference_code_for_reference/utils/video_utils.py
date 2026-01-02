import cv2
import subprocess
import shlex
import json
import os
from config.settings import *
from utils.log_utils import logger
import requests
import re

def get_video_resolution_and_fps(source_path):
    """
    Uses ffprobe to get video width, height, and FPS from a video file or stream URL.

    Args:
        source_path (str): Local file path or video stream URL (.mp4 or .m3u8)

    Returns:
        tuple: (width, height, fps_float) or (None, None, None) if error occurs
    """
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate ' \
              f'-of default=noprint_wrappers=1 "{source_path}"'
        output = subprocess.check_output(shlex.split(cmd)).decode().strip().split()
        width = int(output[0].split('=')[1])
        height = int(output[1].split('=')[1])
        fps_str = output[2].split('=')[1]  # e.g. "30000/1001"
        num, denom = map(int, fps_str.split('/'))
        fps = round(num / denom, 6) if denom != 0 else 0.0  # return as float rounded to 2 decimals
        return width, height, fps
    except Exception as e:
        print(f"Error while getting resolution and FPS for {source_path}: {e}")
        return None, None, None

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def get_duration_from_m3u8(m3u8_url):
    """Get total duration by parsing m3u8 playlist segments"""
    try:
        response = requests.get(m3u8_url)
        content = response.text
        
        # Find all segment durations
        segments = re.findall(r'#EXTINF:([0-9.]+),', content)
        total_duration = sum(float(duration) for duration in segments)
        
        print(f"Found {len(segments)} segments")
        print(f"Total duration from m3u8: {total_duration:.2f} seconds")
        return total_duration
    except Exception as e:
        print(f"Error parsing m3u8: {e}")
        return None

def get_video_info(m3u8_url):
    """Get video frame rate and total duration"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_entries', 'stream=r_frame_rate,nb_frames,duration',
        m3u8_url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    print(f"data is: {data}")
    
    stream = data['streams'][0]
    print(f"stream is: {stream}")
    frame_rate = eval(stream['r_frame_rate'])  # e.g., "30/1" -> 30.0
    duration = get_duration_from_m3u8(m3u8_url)
    total_frames = int(duration * frame_rate)
    
    return frame_rate, total_frames, duration

def frame_to_time(frame_num, frame_rate):
    """Convert frame number to time in seconds"""
    return frame_num / frame_rate

def time_to_frame(time_seconds, frame_rate):
    """Convert time to frame number"""
    return int(time_seconds * frame_rate)

