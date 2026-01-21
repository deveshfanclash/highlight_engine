import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_path)
import time
from queue import Empty
import boto3
from decimal import Decimal
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from utils import logger
from config.settings import *
from pathlib import Path
from datetime import datetime
from datetime import datetime

ENV_PATH = os.path.join(root_path ,  ".env")
load_dotenv(dotenv_path=ENV_PATH)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_DB")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY_DB")
AWS_REGION = os.getenv("AWS_REGION")
TABLE_NAME_VIDEO_FRAME_METADATA = env_cfg.TABLE_NAME_VIDEO_FRAMES_METADATA


def inference_writer_worker_db(output_q, match_id):
    WRITE_DELAY = 12
    WRITE_INTERVAL = 0.250

    records = []
    last_write = time.time()

    # Setup DynamoDB
    dynamodb = boto3.resource(
        'dynamodb',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    table = dynamodb.Table(env_cfg.TABLE_NAME_OBJECT_INFERENCE)

    try:
        while True:
            try:
                data = output_q.get(timeout=5)
                if data is None:
                    print("[INFO] Received termination signal for writer.")
                    break

                records.append(data)

                if len(records) >= WRITE_DELAY or (time.time() - last_write) > WRITE_INTERVAL:
                    for record in records:
                        try:
                            frame_number = int(record["frame_number"])
                            
                            # Check if this is cricket merged result
                            if "cricket_model_boxes" in record and "person_head_model_boxes" in record and "person_detection_model_boxes" in record:
                                # Cricket triple model result - write as single record (no race condition)
                                cricket_bboxes = [[round(float(x), 2) for x in box] for box in record["cricket_model_boxes"]]
                                cricket_confidence = [round(float(c), 2) for c in record["cricket_model_confidences"]]
                                cricket_class_id = [int(cid) for cid in record["cricket_model_classes"]]
                                
                                person_head_bboxes = [[round(float(x), 2) for x in box] for box in record["person_head_model_boxes"]]
                                person_head_confidence = [round(float(c), 2) for c in record["person_head_model_confidences"]]
                                person_head_class_id = [int(cid) for cid in record["person_head_model_classes"]]
                                
                                person_detection_bboxes = [[round(float(x), 2) for x in box] for box in record["person_detection_model_boxes"]]
                                person_detection_confidence = [round(float(c), 2) for c in record["person_detection_model_confidences"]]
                                person_detection_class_id = [int(cid) for cid in record["person_detection_model_classes"]]

                                item = {
                                    "match_id": match_id,
                                    "frame_number": frame_number,
                                    "cricket_model_bbox": [[Decimal(str(x)) for x in box] for box in cricket_bboxes],
                                    "cricket_model_confidence": [Decimal(str(c)) for c in cricket_confidence],
                                    "cricket_model_class_id": cricket_class_id,
                                    "person_head_model_bbox": [[Decimal(str(x)) for x in box] for box in person_head_bboxes],
                                    "person_head_model_confidence": [Decimal(str(c)) for c in person_head_confidence],
                                    "person_head_model_class_id": person_head_class_id,
                                    "person_detection_model_bbox": [[Decimal(str(x)) for x in box] for box in person_detection_bboxes],
                                    "person_detection_model_confidence": [Decimal(str(c)) for c in person_detection_confidence],
                                    "person_detection_model_class_id": person_detection_class_id
                                }
                            else:
                                # Single model result (Football/Volleyball)
                                bboxes = [[round(float(x), 2) for x in box] for box in record["boxes"]]
                                confidence = [round(float(c), 2) for c in record["confidences"]]
                                class_id = [int(cid) for cid in record["classes"]]

                                item = {
                                    "match_id": match_id,
                                    "frame_number": frame_number,
                                    "bbox": [[Decimal(str(x)) for x in box] for box in bboxes],
                                    "confidence": [Decimal(str(c)) for c in confidence],
                                    # "class_id": class_id
                                    "class_id": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                                }
                            
                            # print(item)
                            if frame_number % 600 == 0:
                                logger.info(f"✅ Realtime Inference write till: frame_number={frame_number}")
                            table.put_item(Item=item)
                            # print(f"[INFO] Wrote record for frame {frame_number} to DynamoDB.")

                        except Exception as e:
                            print(f"[ERROR] Failed to write record to DynamoDB: {e}")

                    records.clear()
                    last_write = time.time()

            except Empty:
                continue

    except Exception as e:
        print(f"[ERROR] Exception in inference_writer_worker_db: {e}")

    finally:
        # ✅ Final flush
        if records:
            try:
                for record in records:
                    frame_number = int(record["frame_number"])
                    
                    # Check if this is cricket merged result
                    if "cricket_model_boxes" in record and "person_head_model_boxes" in record and "person_detection_model_boxes" in record:
                        # Cricket triple model result - write as single record (no race condition)
                        cricket_bboxes = [[round(float(x), 2) for x in box] for box in record["cricket_model_boxes"]]
                        cricket_confidence = [round(float(c), 2) for c in record["cricket_model_confidences"]]
                        cricket_class_id = [int(cid) for cid in record["cricket_model_classes"]]
                        
                        person_head_bboxes = [[round(float(x), 2) for x in box] for box in record["person_head_model_boxes"]]
                        person_head_confidence = [round(float(c), 2) for c in record["person_head_model_confidences"]]
                        person_head_class_id = [int(cid) for cid in record["person_head_model_classes"]]
                        
                        person_detection_bboxes = [[round(float(x), 2) for x in box] for box in record["person_detection_model_boxes"]]
                        person_detection_confidence = [round(float(c), 2) for c in record["person_detection_model_confidences"]]
                        person_detection_class_id = [int(cid) for cid in record["person_detection_model_classes"]]

                        item = {
                            "match_id": match_id,
                            "frame_number": frame_number,
                            "cricket_model_bbox": [[Decimal(str(x)) for x in box] for box in cricket_bboxes],
                            "cricket_model_confidence": [Decimal(str(c)) for c in cricket_confidence],
                            "cricket_model_class_id": cricket_class_id,
                            "person_head_model_bbox": [[Decimal(str(x)) for x in box] for box in person_head_bboxes],
                            "person_head_model_confidence": [Decimal(str(c)) for c in person_head_confidence],
                            "person_head_model_class_id": person_head_class_id,
                            "person_detection_model_bbox": [[Decimal(str(x)) for x in box] for box in person_detection_bboxes],
                            "person_detection_model_confidence": [Decimal(str(c)) for c in person_detection_confidence],
                            "person_detection_model_class_id": person_detection_class_id
                        }
                    else:
                        # Single model result (Football/Volleyball)
                        bboxes = [[round(float(x), 2) for x in box] for box in record["boxes"]]
                        confidence = [round(float(c), 2) for c in record["confidences"]]
                        class_id = [int(cid) for cid in record["classes"]]

                        item = {
                            "match_id": match_id,
                            "frame_number": frame_number,
                            "bbox": [[Decimal(str(x)) for x in box] for box in bboxes],
                            "confidence": [Decimal(str(c)) for c in confidence],
                            # "class_id": class_id
                            "class_id": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                        }
                    
                    # print(item)
                    table.put_item(Item=item)

                print(f"[INFO] Final flush: wrote {len(records)} records to DynamoDB.")
            except Exception as e:
                print(f"[ERROR] Failed to write final records: {e}")

        print("[INFO] Writer terminated gracefully.")




def upload_camera_changes_to_dynamo(change_events, match_id):
 
    try:
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        table = dynamodb.Table(env_cfg.TABLE_NAME_CAMERA_CHANGE)
        try:
            with table.batch_writer(overwrite_by_pkeys=["match_id", "frame_number"]) as writer:
                for event in change_events:
                    item = {
                        "match_id": match_id,
                        "frame_number": int(event["frame_number"])  # ✅ this is the frame number = idx - 1
                    }
                    writer.put_item(Item=item)
        except Exception as e:
            print(str(e))
        print(f"[INFO] Uploaded {len(change_events)} camera change events for match: {match_id}")
        logger.info(f"[INFO] Uploaded {len(change_events)} camera change events for match: {match_id}")
    except Exception as e:
        print(f"[ERROR] Failed to upload camera changes to DynamoDB: {e}")
        logger.error(f"[ERROR] Failed to upload camera changes to DynamoDB: {e}")

def write_frames_metadata_to_dynamo(batch_items):
    """
    Extract keyframe metadata from a .ts file and write all entries to DynamoDB in a single batch operation.
    
    Args:
        match_id (str): Unique identifier for the match
        ts_path (str): Path to the .ts file
        frame_number (int): Starting frame number
    
    Returns:
        int: Updated frame number after processing all keyframes
    """
    try:
        # Setup DynamoDB
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        table = dynamodb.Table(TABLE_NAME_VIDEO_FRAME_METADATA)

        # Write all items in a single batch operation
        if batch_items:
            with table.batch_writer(overwrite_by_pkeys=["match_id", "ptstime"]) as writer:
                for item in batch_items:
                    # print(item)
                    writer.put_item(Item=item)

            print(f"[INFO] Successfully wrote {len(batch_items)} frame entries to DynamoDB")
        else:
            print(f"[WARNING] No keyframes found")
        return None

    except ClientError as e:
        print(f"[ERROR] DynamoDB ClientError{e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to process frames {e}")
        return None



def main():
    pass


if __name__=='__main__':
    main()