# Standard Library
import os
import sys
import time
import select
import logging
import traceback
import subprocess
from pathlib import Path
from threading import Thread
import multiprocessing as mp
from multiprocessing import Process, Queue
from queue import Empty
from logging.handlers import RotatingFileHandler
from decimal import Decimal

# Third-Party Libraries
import torch
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.settings import *
from database.dynamodb_writer import inference_writer_worker_db
from utils import copy_csv_src_to_dest, frame_to_timecode, get_video_resolution_and_fps, get_best_stream_url
from utils.resource_utils import download_file_from_url
from database.data_ingestion import get_video_frame_metadata_v2
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


def cricket_model_process_worker(model_path, input_queue, output_queue, device, model_name):
    """Persistent worker process for cricket model inference - loads model once and processes all batches"""
    try:
        # Load model once in this process
        print(f"[INFO] Loading {model_name} model in process...")
        model = YOLO(model_path)
        print(f"[INFO] {model_name} model loaded successfully in process")
        
        while True:
            try:
                # Wait for batch from input queue
                batch_data = input_queue.get(timeout=1)
                if batch_data is None:  # Termination signal
                    print(f"[INFO] {model_name} process received termination signal")
                    break
                
                frames_batch, batch_numbers, batch_id = batch_data
                
                # Run inference
                results = model.predict(frames_batch, device=device, verbose=False)
                
                # Process results
                processed_results = []
                for res in results:
                    processed_results.append({
                        'boxes': [[round(val, 2) for val in box] for box in res.boxes.xyxy.cpu().numpy().tolist()],
                        'confidences': [round(conf, 2) for conf in res.boxes.conf.cpu().numpy().tolist()],
                        'classes': [round(cls, 2) for cls in res.boxes.cls.cpu().numpy().tolist()]
                    })
                
                # Send results back with batch_id for synchronization
                output_queue.put((model_name, processed_results, batch_numbers, batch_id))
                
            except Exception as e:
                print(f"[ERROR] Exception in {model_name} process: {e}")
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"[ERROR] Critical error in {model_name} process: {e}")
    finally:
        print(f"[INFO] {model_name} process shutting down")

def inference_worker(input_q, output_q, batch_size, device, game_id, model_path, IS_YOLO_DEFAULT_MODEL_USE=False):
    try:
        # Initialize based on game type
        if game_id == "68b7fd9f414719f5a368fae0" and os.getenv('IS_CRICKET_MERGED_MODEL', 'True').lower() == 'false':  # Cricket Model - 3 separate models only
            # Use 3 separate cricket models (existing logic)
            cricket_model_path = os.getenv('CRICKET_MODEL_PATH')
            person_head_model_path = os.getenv('PERSON_HEAD_MODEL_PATH')
            person_detection_model_path = os.getenv('DEFAULT_YOLO_MODEL_PATH')
            
            if not cricket_model_path or not person_head_model_path or not person_detection_model_path:
                logger.error("Cricket models not available")
                raise ValueError("Cricket models not available")
            
            # Create persistent processes for cricket models
            cricket_input_queue = Queue()
            cricket_output_queue = Queue()
            person_head_input_queue = Queue()
            person_head_output_queue = Queue()
            person_detection_input_queue = Queue()
            person_detection_output_queue = Queue()
            
            # Start persistent cricket model process
            cricket_process = Process(
                target=cricket_model_process_worker,
                args=(cricket_model_path, cricket_input_queue, cricket_output_queue, device, 'cricket_model')
            )
            cricket_process.start()
            
            # Start persistent person head model process
            person_head_process = Process(
                target=cricket_model_process_worker,
                args=(person_head_model_path, person_head_input_queue, person_head_output_queue, device, 'person_head_model')
            )
            person_head_process.start()
            
            # Start persistent person detection model process
            person_detection_process = Process(
                target=cricket_model_process_worker,
                args=(person_detection_model_path, person_detection_input_queue, person_detection_output_queue, device, 'person_detection_model')
            )
            person_detection_process.start()
            
            print(f"[INFO] Cricket persistent processes started (3 models)")
            is_cricket = True
            model = None
        else:  # Single model processing (Football/Volleyball/Cricket Merged)
            pass
            model = YOLO(model_path)
            is_cricket = False
        
        batch_frames, batch_numbers = [], []
        batch_id_counter = 0  # For synchronization
        
        while True:
            try:
                item = input_q.get(timeout=1)  # Wait for frame
            except Empty:
                print("[WARNING] Input queue is empty, retrying...")
                continue

            if item is None:
                print("[INFO] Received termination signal.")
                break

            frame_number, frame = item
            batch_frames.append(frame)
            batch_numbers.append(frame_number)

            # Once enough frames are collected, process batch
            if len(batch_frames) == batch_size:
                if is_cricket:
                    # Cricket 3 separate models processing with SYNCHRONIZED results
                    try:
                        current_batch_id = batch_id_counter
                        batch_id_counter += 1
                        
                        # Send batch to all three persistent processes with batch_id
                        cricket_input_queue.put((batch_frames, batch_numbers, current_batch_id))
                        person_head_input_queue.put((batch_frames, batch_numbers, current_batch_id))
                        person_detection_input_queue.put((batch_frames, batch_numbers, current_batch_id))
                        
                        # Wait for results from all three processes with SYNCHRONIZATION
                        cricket_results_received = False
                        person_head_results_received = False
                        person_detection_results_received = False
                        cricket_name, cricket_results, cricket_batch_numbers, cricket_batch_id = None, None, None, None
                        person_head_name, person_head_results, person_head_batch_numbers, person_head_batch_id = None, None, None, None
                        person_detection_name, person_detection_results, person_detection_batch_numbers, person_detection_batch_id = None, None, None, None
                        
                        # Collect results with batch_id verification
                        while not (cricket_results_received and person_head_results_received and person_detection_results_received):
                            # Check cricket results
                            if not cricket_results_received and not cricket_output_queue.empty():
                                cricket_name, cricket_results, cricket_batch_numbers, cricket_batch_id = cricket_output_queue.get(timeout=1)
                                if cricket_batch_id == current_batch_id:
                                    cricket_results_received = True
                                    # print(f"[INFO] Cricket results received for batch {current_batch_id}")
                                else:
                                    # print(f"[WARNING] Cricket results out of sync: expected {current_batch_id}, got {cricket_batch_id}")
                                    # Put back the wrong result and continue
                                    cricket_output_queue.put((cricket_name, cricket_results, cricket_batch_numbers, cricket_batch_id))
                            
                            # Check person head results
                            if not person_head_results_received and not person_head_output_queue.empty():
                                person_head_name, person_head_results, person_head_batch_numbers, person_head_batch_id = person_head_output_queue.get(timeout=1)
                                if person_head_batch_id == current_batch_id:
                                    person_head_results_received = True
                                    # print(f"[INFO] Person head results received for batch {current_batch_id}")
                                else:
                                    # print(f"[WARNING] Person head results out of sync: expected {current_batch_id}, got {person_head_batch_id}")
                                    # Put back the wrong result and continue
                                    person_head_output_queue.put((person_head_name, person_head_results, person_head_batch_numbers, person_head_batch_id))
                            
                            # Check person detection results
                            if not person_detection_results_received and not person_detection_output_queue.empty():
                                person_detection_name, person_detection_results, person_detection_batch_numbers, person_detection_batch_id = person_detection_output_queue.get(timeout=1)
                                if person_detection_batch_id == current_batch_id:
                                    person_detection_results_received = True
                                    # print(f"[INFO] Person detection results received for batch {current_batch_id}")
                                else:
                                    # print(f"[WARNING] Person detection results out of sync: expected {current_batch_id}, got {person_detection_batch_id}")
                                    # Put back the wrong result and continue
                                    person_detection_output_queue.put((person_detection_name, person_detection_results, person_detection_batch_numbers, person_detection_batch_id))
                            
                            # If we're still waiting, sleep briefly
                            if not (cricket_results_received and person_head_results_received and person_detection_results_received):
                                time.sleep(0.01)
                        
                        # Verify all three results are from the same batch
                        if cricket_batch_id == person_head_batch_id == person_detection_batch_id == current_batch_id:
                            # print(f"[INFO] All three results synchronized for batch {current_batch_id}")
                            
                            # Filter person detection results to only include person class (class_id = 0)
                            filtered_person_detection_results = []
                            for i in range(len(person_detection_results)):
                                filtered_boxes = []
                                filtered_confidences = []
                                filtered_classes = []
                                
                                for j, class_id in enumerate(person_detection_results[i]['classes']):
                                    if int(class_id) == 0:  # Only person class
                                        filtered_boxes.append(person_detection_results[i]['boxes'][j])
                                        filtered_confidences.append(person_detection_results[i]['confidences'][j])
                                        filtered_classes.append(person_detection_results[i]['classes'][j])
                                
                                filtered_person_detection_results.append({
                                    'boxes': filtered_boxes,
                                    'confidences': filtered_confidences,
                                    'classes': filtered_classes
                                })
                            
                            # Merge results for each frame and write as single record
                            for i in range(len(batch_frames)):
                                merged_result = {
                                    'frame_number': batch_numbers[i],
                                    'cricket_model_boxes': cricket_results[i]['boxes'],
                                    'cricket_model_confidences': cricket_results[i]['confidences'],
                                    'cricket_model_classes': cricket_results[i]['classes'],
                                    'person_head_model_boxes': person_head_results[i]['boxes'],
                                    'person_head_model_confidences': person_head_results[i]['confidences'],
                                    'person_head_model_classes': person_head_results[i]['classes'],
                                    'person_detection_model_boxes': filtered_person_detection_results[i]['boxes'],
                                    'person_detection_model_confidences': filtered_person_detection_results[i]['confidences'],
                                    'person_detection_model_classes': filtered_person_detection_results[i]['classes']
                                }
                                output_q.put(merged_result)
                        else:
                            print(f"[ERROR] Batch ID mismatch: cricket={cricket_batch_id}, person_head={person_head_batch_id}, person_detection={person_detection_batch_id}, expected={current_batch_id}")
                            # Fallback: send empty results
                            for i in range(len(batch_frames)):
                                output_q.put({
                                    'frame_number': batch_numbers[i],
                                    'cricket_model_boxes': [],
                                    'cricket_model_confidences': [],
                                    'cricket_model_classes': [],
                                    'person_head_model_boxes': [],
                                    'person_head_model_confidences': [],
                                    'person_head_model_classes': [],
                                    'person_detection_model_boxes': [],
                                    'person_detection_model_confidences': [],
                                    'person_detection_model_classes': []
                                })
                                
                    except Exception as e:
                        print(f"[ERROR] Exception in cricket triple model processing: {e}")
                        # Fallback: send empty results
                        for i in range(len(batch_frames)):
                            output_q.put({
                                'frame_number': batch_numbers[i],
                                'cricket_model_boxes': [],
                                'cricket_model_confidences': [],
                                'cricket_model_classes': [],
                                'person_head_model_boxes': [],
                                'person_head_model_confidences': [],
                                'person_head_model_classes': [],
                                'person_detection_model_boxes': [],
                                'person_detection_model_confidences': [],
                                'person_detection_model_classes': []
                            })
                else:
                    # Single model processing (Football/Volleyball/Cricket Merged)
                    if IS_YOLO_DEFAULT_MODEL_USE: # Hammer Through Model
                        results = model.predict(batch_frames, device=device, verbose=False, classes=[0, 17, 32])
                        #NOTE: here i want to map 0 as 1 , 17 as 2, 32 as 0, and class names = {0: person, 17 : horse, 32 : ball}
                        class_mapping = {0: 1, 17: 2, 32: 0}
                    else:
                        results = model.predict(batch_frames, device=device, verbose=False)
                        class_mapping = {0: 0, 1: 1, 2: 2}
                    for i, res in enumerate(results):
                        output_q.put({
                            'frame_number': batch_numbers[i],
                            'boxes': [[round(val, 2) for val in box] for box in res.boxes.xyxy.cpu().numpy().tolist()],
                            'confidences': [round(conf, 2) for conf in res.boxes.conf.cpu().numpy().tolist()],
                            'classes': [round(class_mapping[int(cls)], 2) for cls in res.boxes.cls.cpu().numpy().tolist()]
                        })

                batch_frames.clear()
                batch_numbers.clear()
    except Exception as e:
        print(f"[ERROR] Exception in inference_worker: {e}")
    finally:
        # Final flush: process any remaining frames
        if batch_frames:
            print(f"[INFO] Processing final batch of {len(batch_frames)} frame(s).")
            try:
                if is_cricket:
                    # Cricket triple model final processing with SYNCHRONIZED results
                    try:
                        current_batch_id = batch_id_counter
                        batch_id_counter += 1
                        
                        # Send final batch to all three persistent processes with batch_id
                        cricket_input_queue.put((batch_frames, batch_numbers, current_batch_id))
                        person_head_input_queue.put((batch_frames, batch_numbers, current_batch_id))
                        person_detection_input_queue.put((batch_frames, batch_numbers, current_batch_id))
                        
                        # Wait for results from all three processes with SYNCHRONIZATION
                        cricket_results_received = False
                        person_head_results_received = False
                        person_detection_results_received = False
                        cricket_name, cricket_results, cricket_batch_numbers, cricket_batch_id = None, None, None, None
                        person_head_name, person_head_results, person_head_batch_numbers, person_head_batch_id = None, None, None, None
                        person_detection_name, person_detection_results, person_detection_batch_numbers, person_detection_batch_id = None, None, None, None
                        
                        # Collect results with batch_id verification
                        while not (cricket_results_received and person_head_results_received and person_detection_results_received):
                            # Check cricket results
                            if not cricket_results_received and not cricket_output_queue.empty():
                                cricket_name, cricket_results, cricket_batch_numbers, cricket_batch_id = cricket_output_queue.get(timeout=1)
                                if cricket_batch_id == current_batch_id:
                                    cricket_results_received = True
                                    print(f"[INFO] Cricket final results received for batch {current_batch_id}")
                                else:
                                    print(f"[WARNING] Cricket final results out of sync: expected {current_batch_id}, got {cricket_batch_id}")
                                    cricket_output_queue.put((cricket_name, cricket_results, cricket_batch_numbers, cricket_batch_id))
                            
                            # Check person head results
                            if not person_head_results_received and not person_head_output_queue.empty():
                                person_head_name, person_head_results, person_head_batch_numbers, person_head_batch_id = person_head_output_queue.get(timeout=1)
                                if person_head_batch_id == current_batch_id:
                                    person_head_results_received = True
                                    print(f"[INFO] Person head final results received for batch {current_batch_id}")
                                else:
                                    print(f"[WARNING] Person head final results out of sync: expected {current_batch_id}, got {person_head_batch_id}")
                                    person_head_output_queue.put((person_head_name, person_head_results, person_head_batch_numbers, person_head_batch_id))
                            
                            # Check person detection results
                            if not person_detection_results_received and not person_detection_output_queue.empty():
                                person_detection_name, person_detection_results, person_detection_batch_numbers, person_detection_batch_id = person_detection_output_queue.get(timeout=1)
                                if person_detection_batch_id == current_batch_id:
                                    person_detection_results_received = True
                                    print(f"[INFO] Person detection final results received for batch {current_batch_id}")
                                else:
                                    print(f"[WARNING] Person detection final results out of sync: expected {current_batch_id}, got {person_detection_batch_id}")
                                    person_detection_output_queue.put((person_detection_name, person_detection_results, person_detection_batch_numbers, person_detection_batch_id))
                            
                            if not (cricket_results_received and person_head_results_received and person_detection_results_received):
                                time.sleep(0.01)
                        
                        # Verify all three results are from the same batch
                        if cricket_batch_id == person_head_batch_id == person_detection_batch_id == current_batch_id:
                            print(f"[INFO] All three final results synchronized for batch {current_batch_id}")
                            
                            # Filter person detection results to only include person class (class_id = 0)
                            filtered_person_detection_results = []
                            for i in range(len(person_detection_results)):
                                filtered_boxes = []
                                filtered_confidences = []
                                filtered_classes = []
                                
                                for j, class_id in enumerate(person_detection_results[i]['classes']):
                                    if int(class_id) == 0:  # Only person class
                                        filtered_boxes.append(person_detection_results[i]['boxes'][j])
                                        filtered_confidences.append(person_detection_results[i]['confidences'][j])
                                        filtered_classes.append(person_detection_results[i]['classes'][j])
                                
                                filtered_person_detection_results.append({
                                    'boxes': filtered_boxes,
                                    'confidences': filtered_confidences,
                                    'classes': filtered_classes
                                })
                            
                            # Merge results for each frame
                            for i in range(len(batch_frames)):
                                merged_result = {
                                    'frame_number': batch_numbers[i],
                                    'cricket_model_boxes': cricket_results[i]['boxes'],
                                    'cricket_model_confidences': cricket_results[i]['confidences'],
                                    'cricket_model_classes': cricket_results[i]['classes'],
                                    'person_head_model_boxes': person_head_results[i]['boxes'],
                                    'person_head_model_confidences': person_head_results[i]['confidences'],
                                    'person_head_model_classes': person_head_results[i]['classes'],
                                    'person_detection_model_boxes': filtered_person_detection_results[i]['boxes'],
                                    'person_detection_model_confidences': filtered_person_detection_results[i]['confidences'],
                                    'person_detection_model_classes': filtered_person_detection_results[i]['classes']
                                }
                                output_q.put(merged_result)
                        else:
                            print(f"[ERROR] Final batch ID mismatch: cricket={cricket_batch_id}, person_head={person_head_batch_id}, person_detection={person_detection_batch_id}, expected={current_batch_id}")
                            
                    except Exception as e:
                        print(f"[ERROR] Failed during cricket final batch processing: {e}")
                else:
                    # Single model processing (Football/Volleyball/Cricket Merged)
                    if game_id in ALL_PERSON_TRACK_GAME: # Hammer Through Model
                        results = model.predict(batch_frames, device=device, verbose=False, classes=[0, 17, 32])
                        #NOTE: here i want to map 0 as 1 , 17 as 2, 32 as 0, and class names = {0: person, 17 : horse, 32 : ball}
                        class_mapping = {0: 1, 17: 2, 32: 0}
                    else:
                        results = model.predict(batch_frames, device=device, verbose=False)
                        class_mapping = {0: 0, 1: 1, 2: 2}
                    for i, res in enumerate(results):
                        output_q.put({
                            'frame_number': batch_numbers[i],
                            'boxes': [[round(val, 2) for val in box] for box in res.boxes.xyxy.cpu().numpy().tolist()],
                            'confidences': [round(conf, 2) for conf in res.boxes.conf.cpu().numpy().tolist()],
                            'classes': [round(class_mapping[int(cls)], 2) for cls in res.boxes.cls.cpu().numpy().tolist()]
                        })
            except Exception as e:
                print(f"[ERROR] Failed during final batch processing: {e}")

        # Cleanup persistent processes for cricket (only for 3 separate models mode)
        if is_cricket and cricket_process is not None:
            try:
                print(f"[INFO] Terminating cricket persistent processes...")
                # Send termination signals
                cricket_input_queue.put(None)
                person_head_input_queue.put(None)
                person_detection_input_queue.put(None)
                
                # Wait for processes to finish
                cricket_process.join(timeout=10)
                person_head_process.join(timeout=10)
                person_detection_process.join(timeout=10)
                
                # Force terminate if still alive
                if cricket_process.is_alive():
                    cricket_process.terminate()
                if person_head_process.is_alive():
                    person_head_process.terminate()
                if person_detection_process.is_alive():
                    person_detection_process.terminate()
                    
                print(f"[INFO] Cricket persistent processes terminated")
            except Exception as e:
                print(f"[ERROR] Error terminating cricket processes: {e}")

        print("[INFO] Inference worker shutting down gracefully.")

def csv_writer_worker(output_q, csv_path, match_id):
    WRITE_DELAY = 12  # Number of frames to buffer before writing
    WRITE_INTERVAL = 0.250  # Time interval threshold in seconds
    
    records = []
    last_write = time.time()

    try:
        while True:
            try:
                data = output_q.get(timeout=5)
                if data is None:
                    print("[INFO] Received termination signal for CSV writer.")
                    break

                records.append(data)

                # Write if enough records or time elapsed
                if len(records) >= WRITE_DELAY or (time.time() - last_write) > WRITE_INTERVAL:
                    df = pd.DataFrame(records)
                    df.to_csv(
                        csv_path,
                        mode='a',
                        header=not os.path.exists(csv_path),
                        index=False
                    )
                    # print(f"[INFO] Wrote {len(records)} records to CSV.")
                    records.clear()
                    last_write = time.time()

            except Empty:
                continue

    except Exception as e:
        print(f"[ERROR] Exception in csv_writer_worker: {e}")

    finally:
        # Final write for any leftover records
        if records:
            try:
                df = pd.DataFrame(records)
                df.to_csv(
                    csv_path,
                    mode='a',
                    header=not os.path.exists(csv_path),
                    index=False
                )
                print(f"[INFO] Final flush: wrote {len(records)} records to CSV.")
            except Exception as e:
                print(f"[ERROR] Failed to write final records: {e}")

        print("[INFO] CSV writer terminated gracefully.")

def get_real_time_frames(message, ):
    """
    Processes an HLS stream to extract frames, compute perceptual hashes, 
    and detect replay sections in a football match.

    Args:
        message (dict): Contains stream URL, match ID, and other metadata.
        width (int): Width of the resized frame (default: quarter of 1920).
        height (int): Height of the resized frame (default: quarter of 1080).
        csv_file (str, optional): Path to the CSV file for metadata storage.
        output_folder (str, optional): Directory to save extracted images.
    """
    hls_url_raw = message.get("streamUrl")
    match_id = message.get("matchId", None)
    # game_name = message.get("game", {"id":"", "name":""}).get("name", "FOOTBALL")
    hls_url = get_best_stream_url(hls_url_raw, "_1080p.m3u8")
    width, height, fps = get_video_resolution_and_fps(hls_url)
    # print("width, height, fps : ", width, height, fps)
    league_name = message.get("league", "nsl")
    lang = message.get("lang", "en")
    input_queues = message.get("queues", None)
    num_queues = len(input_queues)
    batch_size = message.get("batch_size", 8)

    # match_id Need to use match id for csv file creation

    if not hls_url:
        logger.warning("No stream URL found in Kafka message.")
        return {}

    # get realtime stream based segment and frame number
    realtime_stream_metadata = get_video_frame_metadata_v2(match_id, hls_url_raw, fps)
    frame_number = realtime_stream_metadata.get("frame_number", 1)  # if process started between match than start from realtime stream
    segment_number = realtime_stream_metadata.get("segment_number", 1)
    print("\n\nreal_time_inference file: realtime_stream_metadata: ", realtime_stream_metadata, "\n\n")

    # NOTE: need to add if stream stop somewhere then start from particular time and need to test also same.
    ffmpeg_cmd = [
        "ffmpeg",
        # "-hwaccel", "cuda",
        # "-c:v", "h264_cuvid", 
        "-live_start_index", str(segment_number-1),
        # '-ss', ":".join(stream_start_time),
        # '-ss', '01:42:00',
        # '-ss', '01:12:00',
        # '-t', '00:00:10',
        "-i",
        hls_url,
        "-hls_flags", "delete_segments",
        "-vf",
        f"scale={width}:{height}",  # Resize
        #  '-vf', f'crop={ int(531*(width/1920)) }:{ int(298*(height/1080)) }:{ int(531*(width/1920)) -  int(1389*(width/1920)) }:{ int(781*(height/1080)) - int(298*(height/1080))}', # int(298*(height/1080)):int(781*(height/1080)), int(531*(width/1920)):int(1389*(width/1920))
        # '-vf', f'showinfo',
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        # '-show_frames',  # Enable frame-level data
        # '-loglevel', 'debug',  # Log all metadata
        "-",
    ]

    logger.info(f"Starting FFmpeg process for HLS stream: {hls_url}")
    try:
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def capture_ffmpeg_logs(proc):
                """Reads and logs FFmpeg stderr output."""
                try:
                    while proc.poll() is None:
                        error_output = proc.stderr.read1(1024)
                        if error_output:
                            decoded_error = error_output.decode("utf-8")
                            if frame_number % 600 == 0:
                                logger.error(f"error in capture_ffmpeg_logs {decoded_error}")
                except Exception as e:
                    logger.error(f"Error reading FFmpeg logs: {str(e)}")

        # Start a separate thread to handle FFmpeg stderr logs
        stderr_thread = Thread(target=capture_ffmpeg_logs, args=(ffmpeg_proc,))
        stderr_thread.start()
        processing_start_time = time.time()
        while True:
            try:
                if frame_number % 500 == 0:
                    
                    logger.info(f"Processing frame: {frame_number} , time : {str(frame_to_timecode(frame_number, fps))}")
                    # logger.info(f"Processing time for system : {time.time() - processing_start_time:.2f} seconds")
                    processing_start_time = time.time()
                # Wait for data availability in stdout (non-blocking)
                rlist, _, _ = select.select([ffmpeg_proc.stdout], [], [], 0.1)
                if rlist:
                    raw_frame = ffmpeg_proc.stdout.read(width * height * 3)  # Read frame data
                else:
                    continue  # Skip iteration if no data is available

                # Check if the stream has ended
                if len(raw_frame) != width * height * 3:
                    logger.info("End of stream detected. Stopping frame processing.")
                    # for i in range(input_queues):
                    #     target_queue = input_queues[i]
                    #     target_queue.put(None)
                    break
                
                # Convert raw frame data to a NumPy array and reshape
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                # Calculate batch ID for the current frame
                batch_id = frame_number // batch_size

                # Assign one queue for the whole batch
                queue_index = batch_id % num_queues

                # Distribute frames evenly to each queue
                target_queue = input_queues[queue_index] # frame_number % num_queues
                # 1. Send frame for real_time_inference
                target_queue.put((frame_number, frame))
                # 2. Send frame for Camera view change identification

                # print("frame_number: ", frame_number)
                frame_number += 1
            except Exception as e:
                logger.error(f"Error processing frame {frame_number}: {str(e)}")
                if frame_number == 1: 
                    break
                continue
    except Exception as e:
        logger.critical(f"Critical error in HLS stream processing: {str(e)}", exc_info=True)
    finally:
        try:
            for q in input_queues:
                q.put(None)
            if ffmpeg_proc:
                ffmpeg_proc.stdout.close()
                ffmpeg_proc.stderr.close()
                # ffmpeg_proc.terminate()
                # ffmpeg_proc.wait()
                
                logger.info("FFmpeg process terminated successfully.")
                # frame_assigner_closer()
        except Exception as e:
            logger.warning(f"Error while closing FFmpeg process: {str(e)}")

        if stderr_thread.is_alive():
            stderr_thread.join()
        logger.info("Processing completed.")

def real_time_inference(message):
    try:
        # System setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 1
        num_workers = 4
        match_id = message.get("matchId", None)
        game_id = message.get("game", {"id":"68a5d0432de1b006dc1ba003", "name":"Football"}).get("id", "68a5d0432de1b006dc1ba003") # get with default value
        # Check Model existance in instance folder if not download from CDN
        ALL_PERSON_TRACK_GAME = ["5f757d4d53f55124cedc8baf", "5f51f4c31493307c241eee01", "60e3f4afe771f332c8f0b299", "613af39b140a39181c6468db", "65bb360d0e406c26d3ecc998", "60ec12b93 3e8f12cf4369e0", "unknown", "66d6c02a3bcddbe90596e219", "66d6e8393bcddbe90596e21b", "670d2535cf4de84701607781", "6718b6b3a1ad42510b0930b9", "6748331f70fd55361942a736", "64f005373ea28abf77106ca3", "67fc6e626d0b02e4b174287b", "681d9c80039aed2d32f6334c", "685297f49e24feb988a1ad4e", "6847f7dad4c69890176bb27c", "68593898ad966fbef19b59ad", "6865556ea425fc77ac37b887", "686f466040b6ff00535658f6", "686f9f9840b6ff0053565929", "6875e08440b6ff0053565966", "68774ad4fb3597dc20397615", "688c1c3b45e318c6b03ad435", "689104c360df09a49d097f4f", "68a54ff32de1b006dc1b9fea", "68adb24385e15fb47c349b7f", "69049b1ddf22c84448a7b9e2", "69049918df22c84448a7b9de", "6904990ddf22c84448a7b9dc", "6904988fdf22c84448a7b9da", "69049880df22c84448a7b9d8", "69049865df22c84448a7b9d6", "6904977ddf22c84448a7b9d4", "6904973cdf22c84448a7b9d2", "6904972ddf22c84448a7b9d0", "690496d4df22c84448a7b9ce", "690496bedf22c84448a7b9cc", "690496a9df22c84448a7b9ca", "6904960cdf22c84448a7b9c8", "690495fcdf22c84448a7b9c6", "69049597df22c84448a7b9c4", "6904957edf22c84448a7b9c2", "690494dadf22c84448a7b9c0", "6904946bdf22c84448a7b9be", "6904936fdf22c84448a7b9bc", "69049307df22c84448a7b9ba", "690492abdf22c84448a7b9b8", "690491c8df22c84448a7b9b4", "69048ec2df22c84448a7b9af"]

        IS_YOLO_DEFAULT_MODEL_USE = False
        if game_id in ALL_PERSON_TRACK_GAME:
            IS_YOLO_DEFAULT_MODEL_USE = True
        # 
        if game_id in ["68695c52a425fc77ac37b8da", "68248cf49d3393de142e16a8", "6904900bdf22c84448a7b9b2", "68b7fd9f414719f5a368fae0", "6622504e845d0e572cddc306", "69049223df22c84448a7b9b6"]: # Football, Cricket, Volleyball, Hockey, Futsal, TT
            model_folder = game_id
            IS_YOLO_DEFAULT_MODEL_USE = False
        else:
            model_folder = "default"
            IS_YOLO_DEFAULT_MODEL_USE = True
        model_path = os.path.join(os.getenv('BASE_MODEL_PATH'), model_folder, "latest.pt")
        if not os.path.exists(model_path):
            logger.error(f"Model Not Available for game: {game_id}")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model_cdn_url = os.path.join(os.getenv('BASE_MODEL_CDN_PATH'), "models", model_folder, "latest.pt")
            download_file_from_url(model_cdn_url, model_path)
            if not os.path.exists(model_path):
                logger.error("Model Not Available for game: ", game_id)
                raise ValueError("Model Not Available for game: ", game_id)

        # FILE_BASEPATH =  os.path.join(os.getenv("PORTRAIT_2_DEBUG_PATH", "./live_stream_processing/data/"), str(match_id))
        # os.makedirs(FILE_BASEPATH, exist_ok=True)
        # csv_path = os.path.join(FILE_BASEPATH, str(match_id)+"_object_detection"+".csv")
        # if os.path.exists(csv_path):
        #     os.remove(csv_path)
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)

        # Create shared queues
        image_queues = [Queue(maxsize=num_workers*4) for _ in range(num_workers)]
        output_queue = Queue(maxsize=num_workers)

        # Pass queues to the reader
        message['queues'] = image_queues
        message['batch_size'] = batch_size

        # Thread and process setup
        reader_thread = Thread(target=get_real_time_frames, args=(message,))
        infer_processes = [
            Process(target=inference_worker, args=(image_queues[i], output_queue, batch_size, device, game_id, model_path, IS_YOLO_DEFAULT_MODEL_USE))
            for i in range(num_workers)
        ]
        writer_thread = Thread(target=inference_writer_worker_db, args=(output_queue, match_id))
        # writer_thread = Thread(target=csv_writer_worker, args=(output_queue, csv_path, match_id))

        start_time = time.time()

        # Start execution
        reader_thread.start()
        for p in infer_processes:
            p.start()
        writer_thread.start()

        # Wait for completion
        reader_thread.join()
        for p in infer_processes:
            p.join()

        # Signal writer thread to stop
        output_queue.put(None)
        writer_thread.join()

        total_time = time.time() - start_time
        print(f"[INFO] Real-time inference completed in {total_time:.2f} seconds.")
        # time.sleep(30)

    except Exception as e:
        print(f"[ERROR] An exception occurred during real-time inference: {e}")

    finally:
        # Cleanup: Terminate any remaining processes
        for p in infer_processes:
            if p.is_alive():
                p.terminate()
        if reader_thread.is_alive():
            reader_thread.join()
        if writer_thread.is_alive():
            writer_thread.join()
        print("[INFO] Cleanup completed.")
        # NOTE: need to copy to EFS csv file for future get 
        # EFS_DATA_PATH = os.path.join(os.getenv("EFS_PORTRAIT_V2_DATA_COMP_PATH", "/mnt/efs/prod_ds_database/portrait_V2_compatible_data"), str(match_id))
        # copy_csv_src_to_dest(csv_path, EFS_DATA_PATH)

def main():
    # Example usage
    # hls_url = "http://127.0.0.1:8000/index.m3u8" # index.m3u8
    # hls_url = "https://highlights-cdn.spectatr.gg/streams/test_06516813-8547-4981-ab3f-539a4bfbd9fc/playlist/video_173441383952355.m3u8" #1 minute testing file
    # hls_url = "https://highlights-cdn.spectatr.gg/streams/test_5359d239-6539-41b7-918a-333d598430e8/playlist/video_test_5359d239-6539-41b7-918a-333d598430e8.m3u8" # 7 minute testing file

    message = {
        "matchId": "68ad7212e02bcdd9011ee6ab",
        "streamUrl": "https://highlights-cdn.spectatr.gg/hls/production/68ad7212e02bcdd9011ee6ab/67f8b5ea6d0b02e4b1742849/segments.m3u8",
        "streamStatus": "started",
        "league": "league",
        "lang": "en",
        "game": {
            "id": "68695c52a425fc77ac37b8da",
            "name": "Volleyball",
        },
        "tournament": {
            "id": "68a6ef0f116ca0108d840934",
            "name": "Northern Super League 2025",
        }
    }

    # ANOC match
    message = {
        "matchId": "6900b9dfb27a3d87f879efc6",
        "streamUrl": "https://highlights-cdn.spectatr.gg/hls/staging/6900b9dfb27a3d87f879efc6/67876a3d86c44d602c6d57c1/segments.m3u8",
        "streamStatus": "started",
        "league": "anoc",
        "lang": "en",
        "game": {
            "id": "69049b1ddf22c84448a7b9e2", # for cricket
            "name": "athletics",
        },
        "tournament": {
            "id": "68a6ef0f116ca0108d840934",
            "name": "Northern Super League 2025",
        }
    }

    real_time_inference(message)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("total time: ", time.time() - start_time)