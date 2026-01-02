import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import csv
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import select
import threading
import numpy as np
import cv2
from PIL import Image
import imagehash
import time
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.settings import *
from database.data_ingestion import get_video_frame_metadata_v2
from database.dynamodb_writer import upload_camera_changes_to_dynamo
from utils import copy_csv_src_to_dest, frame_to_timecode, get_video_resolution_and_fps, get_best_stream_url

load_dotenv()

log_file = env_cfg.LOG_FILE_BASE_PATH/ "realtime_main.log"

# Configure logging with rotation (max 50MB per file, keep last 3 backups)
logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=3),  # Rotate at 50MB, keep 3 old logs
        logging.StreamHandler()  # Log to console
    ],
)

logger = logging.getLogger(__name__)
logger.info("Logger initialized successfully.")

def write_change_record(w_ch, f_ch, change_id, idx, tc, sim):
    try:
        w_ch.writerow([change_id, idx, tc, sim])
        f_ch.flush()
    except Exception as e:
        print(f"[ERROR] Writing change record failed: {e}")

# Write a frame-to-segment mapping entry

def write_map_record(w_map, f_map, idx, change_id):
    try:
        w_map.writerow([idx, change_id])
        f_map.flush()
    except Exception as e:
        print(f"[ERROR] Writing map record failed: {e}")

# Compute perceptual hash and histogram from a frame
def compute_frame_features(frame_bgr, width, height):
    try:
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != (width, height):
            frame_bgr = cv2.resize(frame_bgr, (width, height))

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1,2], None, [50,50,50], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ph = imagehash.phash(Image.fromarray(rgb))

        return hist, ph
    except Exception as e:
        print(f"[ERROR] compute_frame_features: {e}")
        return None, None

# Read frames from a local video file and apply frame processor

def process_video_file(source, width, height, process_frame):
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[ERROR] Could not open video file.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (width, height))
            process_frame(resized_frame)
        cap.release()
    except Exception as e:
        print(f"[ERROR] process_video_file: {e}")

# Stream and decode raw video using FFmpeg and process frames

def process_video_stream(source, width, height, process_frame, segment_number=1):
    try:
        source = get_best_stream_url(source, "_1080p.m3u8")
        #NOTE: need to make 1080p stream 
        cmd = [
            "ffmpeg", 
            "-live_start_index", str(segment_number-1),
            # '-ss', '01:42:00',
            # "-hwaccel", "cuda",
            # "-c:v", "h264_cuvid", 
            "-i", source, 
            "-vf", f"scale={width}:{height}", 
            "-f", "rawvideo", 
            "-pix_fmt", 
            "bgr24", 
            "-"
            ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def log_err():
            try:
                while proc.poll() is None:
                    e = proc.stderr.read1(1024)
                    if e:
                        print(e.decode("utf-8", errors="ignore"))
            except Exception as e:
                print(f"[ERROR] log_err: {e}")

        threading.Thread(target=log_err, daemon=True).start()

        frame_size = width * height * 3
        while True:
            r, _, _ = select.select([proc.stdout], [], [], 0.1)
            if not r:
                continue
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
            process_frame(frame)

        proc.stdout.close()
        proc.stderr.close()
        proc.wait()
    except Exception as e:
        print(f"[ERROR] process_video_stream: {e}")

# Main function to perform inline camera cut detection and write to CSV
def camera_view_stream(message):
    phash_threshold=20
    hist_duplicate_thresh=0.90
    source = message.get("streamUrl", "/home/ubuntu/projects/fb_model_engine_staging/football_analysis/data/input_video_test/small_test_d.mp4")
    width, height, fps = get_video_resolution_and_fps(source)
    if None in (width, height, fps):
        print("[ERROR] Failed to retrieve video metadata.")
        logger.error(f"[ERROR] Failed to retrieve video metadata for camera change.")
        return
    match_id = message.get("matchId", None)

    try:

        # Internal state tracking for detection
        prev_phash = None
        prev_hist = None
        # change_id = 1
        idx = 1
        MIN_FRAME_DIFF = int(fps)
        last_logged_frame = -MIN_FRAME_DIFF

        # Core processing for each frame
        def process_frame(frame_bgr):
            # return
            nonlocal prev_phash, prev_hist, idx, last_logged_frame#, change_id
            try:
                hist, ph = compute_frame_features(frame_bgr, width, height)
                
                # hist, ph = compute_frame_features_gpu(frame_bgr, width, height)
                # return
                if hist is None or ph is None:
                    return
                if prev_phash is not None:
                    sim = prev_phash - ph
                    corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    if sim >= phash_threshold and (corr < hist_duplicate_thresh or np.isnan(corr)):
                        if change_id == 0 or idx - last_logged_frame >= MIN_FRAME_DIFF:
                            change_id += 1
                            tc = frame_to_timecode(idx, fps)
                            # write_change_record(w_ch, f_ch, change_id, idx-1, tc, sim)
                            upload_camera_changes_to_dynamo(
                                [{"frame_number": idx-1}],
                                match_id
                            )
                            print(f"✅ Logged cut: ChangeID={change_id}, FrameIndex={idx}, Timecode={tc}, Similarity={sim}")
                            logger.info(f"✅ Logged cut: ChangeID={change_id}, FrameIndex={idx}, Timecode={tc}, Similarity={sim}")
                            last_logged_frame = idx
                # write_map_record(w_map, f_map, idx-1, change_id)
                # print(f"✅ Logged cut: idx={idx}, change_id={change_id} ")
                prev_phash = ph
                prev_hist = hist
                idx += 1
            except Exception as e:
                print(f"[ERROR] Exception during frame processing: {e}")

        width = width//2
        height = height//2
        # Dispatch to local or streamed video path
        if source.lower().endswith(".mp4") or os.path.isfile(source):
            process_video_file(source, width, height, process_frame)
        else:
            # get realtime stream based segment and frame number
            realtime_stream_metadata = get_video_frame_metadata_v2(match_id, source, fps)
            frame_number = realtime_stream_metadata.get("frame_number", 1)  # if process started between match than start from realtime stream
            segment_number = realtime_stream_metadata.get("segment_number", 1)
            idx = frame_number
            print("\n\ncamera view detection file : ", realtime_stream_metadata, "\n\n")
            process_video_stream(source, width, height, process_frame, segment_number)

    except Exception as e:
        print(f"[ERROR] camera_view_stream failed: {e}")
        logger.error(f"[ERROR] camera_view_stream failed: {e}")
    finally:
        pass
        try:
            pass
            # f_ch.close()
            # f_map.close()
            # EFS_DATA_PATH = os.path.join(os.getenv("EFS_PORTRAIT_V2_DATA_COMP_PATH", "/mnt/efs/prod_ds_database/portrait_V2_compatible_data"), str(match_id))
            # copy_csv_src_to_dest(csv_changes_path, EFS_DATA_PATH)
            # copy_csv_src_to_dest(csv_map_path, EFS_DATA_PATH)
        except Exception as e:
            print(f"[ERROR] Closing file handles: {e}")
            logger.error(f"[ERROR] Closing file handles: {e}")

# Entry point
def main():
    try:
        start = time.time()
        filename = "Musiala_54" # small-test-d
        message = {
            "matchId": filename,
            # "streamUrl": "/Users/vaibhav/Downloads/replaywipes.mp4",
            # "streamUrl": "/Users/vaibhav/Downloads/output_clip_GOAL_25_APR_2.mp4",
            "streamUrl": "https://highlights-cdn.spectatr.gg/hls/production/67f8a857836c9a57ecb69e62/67f8a71e6d0b02e4b1742839/segments.m3u8",
            "streamUrl": "/Users/vaibhav/Desktop/football_projects/fb_model_engine/football_analysis/data/input_video/small-test-d.mp4", # small-test-d Musiala_18
            "streamUrl": "/home/ubuntu/projects/fb_model_engine_staging/football_analysis/data/input_video_test/"+filename+".mp4",
            # "streamUrl": "https://highlights-cdn.spectatr.gg/streams/test_06516813-8547-4981-ab3f-539a4bfbd9fc/playlist/video_173441383952355.m3u8",
            # "streamUrl": "https://highlights-cdn.spectatr.gg/hls/production/67f8a858836c9a57ecb69e81/680e3c7b95718713fe035e43/segments.m3u8",
            "streamStatus": "started",
            "league":"nsl",
            "lang":"fr",
        }

        camera_view_stream(message)
        print(f"Total time: {time.time() - start:.2f} seconds")
    except Exception as e:
        print(f"[ERROR] main: {e}")

if __name__ == "__main__":
    main()