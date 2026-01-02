from enum import Enum
import os
from pathlib import Path
from pydantic import BaseModel, model_validator
from typing import Optional, List, Tuple
from dataclasses import dataclass
import re
print(os.getenv("DEBUG_BASE_PATH_DEVELOPMENT"))

class Environment(Enum):
    LOCAL = "local"
    PRODUCTION = "production"
    DEVELOPMENT_TESTING = "development_testing"

class GameType(Enum):
    FOOTBALL = "football"
    CRICKET = "cricket"
    VOLLEYBALL = "volleyball"
    BADMINTON = "badminton"
    TABLE_TENNIS = "table_tennis"
    ICE_HOCKEY = "ice_hockey"
    BASKETBALL = "basketball"

class ModelName(str, Enum):
    RF_DETR = "rf_detr"
    RF_DETR_BASE = "rf_detr_base"
    RF_DETR_LARGE = "rf_detr_large"
    YOLO_8 = "yolov8n"
    YOLO_12 = "yolov12n"

class ModelType(str, Enum):
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    KEY_POINT_DETECTION = "keypoint_detection"


class TargetClass(str, Enum):
    PERSON = "person"
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    THROW_IN = "throw_in"

class InferenceConfig(BaseModel):
    """Configuration for inference"""
    model_type: Optional[str] =None
    model_path: Optional[str] = None
    confidence_threshold: float = 0.5
    max_detections: Optional[int] = 1
    target_classes: List[TargetClass] = [TargetClass.FOOTBALL, TargetClass.PERSON]
    save_annotated_video: bool = False
    output_dir: Optional[str] = None
    custom_model: Optional[bool] = True

class DetectionResult(BaseModel):
    """Structure for detection results"""
    frame_id: int
    timestamp: float
    bbox: Tuple[float, float, float, float]
    class_name: str
    confidence: float
    prediction_time: float

#---------------------------------Environment Config----------------------------
class AppConfig:
    #Constants
    IS_LOCAL = False

    #URLS config
    REPOSITORY_CDN_BASE = "https://repo-cdn.spectatr.gg"
    HIGHLIGHTS_CDN_BASE = "https://highlights-cdn.spectatr.gg"
    HIGHLIGHTS_S3_BUCKET = "https://highlights-storage.s3.us-east-2.amazonaws.com"
    # AWS metadata endpoints
    AWS_IMDSV2_TOKEN = "http://169.254.169.254/latest/api/token"
    AWS_INSTANCE_ID = "http://169.254.169.254/latest/meta-data/instance-id"

    #AWS Config
    AWS_REGION: str = os.getenv("AWS_REGION")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_MAIN")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_MAIN")

    AWS_ACCESS_KEY_DB = os.getenv("AWS_ACCESS_KEY_DB")
    AWS_SECRET_KEY_DB = os.getenv("AWS_SECRET_KEY_DB")

    #S3 Config
    REPOSITORY_S3_BUCKET: str = "repository-storage-mum"
    DS_BUCKET_NAME: str = "highlights-storage"
    # DS_BUCKET_NAME: str = "fc-ds-test-bucket"
    DS_S3_BUCKET = os.getenv("DS_S3_BUCKET", "highlights-storage")
    
class CommonConfig:
    LOG_FILE_PATH = Path.cwd()/f'logs/{os.getenv("LOG_FILE_NAME")}.log'
    LOG_FILE_BASE_PATH = Path.cwd()/f'logs/'
    LOG_GROUP_NAME = os.getenv("LOG_GROUP_NAME")
    TS_FILE_BASE_PATH = os.getenv("TS_FILE_BASE_PATH")
#------------------------------------------------------------------------------------------

def extract_region_from_sqs_url(sqs_url: str) -> str:
    match = re.search(r"https://sqs\.([a-z0-9-]+)\.amazonaws\.com", sqs_url)
    return match.group(1) if match else None

class ProdConfig(AppConfig, CommonConfig):
    DEBUG_BASE_PATH = os.getenv("DEBUG_BASE_PATH_PRODUCTION")
    RETRAIN_DATA_PATH = os.getenv("RETRAIN_DATA_PATH_PRODUCTION")
    SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL_PRODUCTION_RECEIVE")
    ASG_NAME = os.getenv("ASG_NAME_PRODUCTION")
    S3_BASE_URL = os.getenv("S3_BASE_URL_PRODUCTION")
    KAFKA_TOPIC_REAL_TIME_RECEIVE = os.getenv("KAFKA_TOPIC_RECEIVE_PRODUCTION_REAL_TIME")
    KAFKA_BROKER_URL_REAL_TIME = os.getenv("KAFKA_BROKER_URL_RECEIVE_PRODUCTION_REAL_TIME")
    SQS_QUEUE_URL_FALLBACK_SEND = os.getenv("SQS_QUEUE_URL_FALLBACK_SEND_PRODUCTION")
    KAFKA_TOPIC_AI_MATCH_STATUS_INFO=os.getenv("KAFKA_TOPIC_AI_MATCH_STATUS_INFO_PRODUCTION")
    # KAFKA_TOPIC_ML = os.getenv("KAFKA_TOPIC_ML_PRODUCTION")
    SQS_QUEUE_URL_SEND = os.getenv("SQS_QUEUE_URL_SEND_PRODUCTION")
    TABLE_NAME_VIDEO_FRAMES_METADATA = os.getenv("TABLE_NAME_VIDEO_FRAMES_METADATA_PRODUCTION")
    TABLE_NAME_OBJECT_INFERENCE = os.getenv("TABLE_NAME_OBJECT_INFERENCE_PRODUCTION") # 'P2_test_2'
    TABLE_NAME_CAMERA_CHANGE = os.getenv("TABLE_NAME_CAMERA_CHANGE_PRODUCTION") # "camera_change_local"
    MONGODB_URI = os.getenv("MONGO_DB_CONNECTION_STRING_PRODUCTION")
    
    @property
    def SQS_AWS_REGION(self):
        return extract_region_from_sqs_url(os.getenv("SQS_QUEUE_URL_PRODUCTION_RECEIVE"))
    @property
    def IS_LOCAL(self):
        return False

class DevConfig(AppConfig, CommonConfig):
    DEBUG_BASE_PATH = os.getenv("DEBUG_BASE_PATH_DEVELOPMENT") #Path.cwd().parent/"data/output"
    RETRAIN_DATA_PATH = os.getenv("RETRAIN_DATA_PATH_DEVELOPMENT")
    SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL_DEVELOPMENT_RECEIVE")
    KAFKA_TOPIC_REAL_TIME_RECEIVE = os.getenv("KAFKA_TOPIC_RECEIVE_STAGING_REAL_TIME")
    KAFKA_BROKER_URL_REAL_TIME = os.getenv("KAFKA_BROKER_URL_RECEIVE_STAGING_REAL_TIME")
    ASG_NAME = os.getenv("ASG_NAME_DEVELOPMENT")
    S3_BASE_URL = os.getenv("S3_BASE_URL_DEVELOPMENT")
    SQS_QUEUE_URL_FALLBACK_SEND = os.getenv("SQS_QUEUE_URL_FALLBACK_SEND_DEVELOPMENT")
    KAFKA_TOPIC_AI_MATCH_STATUS_INFO=os.getenv("KAFKA_TOPIC_AI_MATCH_STATUS_INFO_STAGING")
    SQS_QUEUE_URL_SEND = os.getenv("SQS_QUEUE_URL_SEND_STAGING")
    TABLE_NAME_VIDEO_FRAMES_METADATA = os.getenv("TABLE_NAME_VIDEO_FRAMES_METADATA_STAGING")
    TABLE_NAME_OBJECT_INFERENCE = os.getenv("TABLE_NAME_OBJECT_INFERENCE_STAGING") # 'P2_test_2'
    TABLE_NAME_CAMERA_CHANGE = os.getenv("TABLE_NAME_CAMERA_CHANGE_STAGING") # "camera_change_local"
    MONGODB_URI = os.getenv("MONGO_DB_CONNECTION_STRING_STAGING")

    @property
    def SQS_AWS_REGION(self):
        return extract_region_from_sqs_url(os.getenv("SQS_QUEUE_URL_DEVELOPMENT_RECEIVE"))
    @property
    def IS_LOCAL(self):
        return False

class LocalConfig(CommonConfig, AppConfig):
    GAMETYPE = GameType.VOLLEYBALL
    DEBUG_BASE_PATH = str(Path(__file__).parent.parent) + '/data'
    print(f"DEBUG_BASE_PATH is {DEBUG_BASE_PATH}")
    """You can give directly the ipnut video path or the input folder path"""
    INPUT_INFERENCE_DATA_PATH = os.getenv("INPUT_INFERENCE_DATA_PATH") if os.getenv("INPUT_INFERENCE_DATA_PATH") else None
    INPUT_VIDEO_FOLDER_PATH = os.getenv("INPUT_VIDEO_FOLDER_PATH") if os.getenv("INPUT_VIDEO_FOLDER_PATH") else None
    TABLE_NAME_VIDEO_FRAMES_METADATA = os.getenv("TABLE_NAME_VIDEO_FRAMES_METADATA_LOCAL")
    TABLE_NAME_OBJECT_INFERENCE = os.getenv("TABLE_NAME_OBJECT_INFERENCE_LOCAL") # 'P2_test_2'
    TABLE_NAME_CAMERA_CHANGE = os.getenv("TABLE_NAME_CAMERA_CHANGE_LOCAL") # "camera_change_local"
    MONGODB_URI = os.getenv("MONGO_DB_CONNECTION_STRING_LOCAL")
    @property
    def IS_LOCAL(self):
        return True
    @property
    def MODEL_PATH(self):
        # if not self.IS_CUSTOM_MODEL:
        #     return Path.cwd().parent/"data/models"
        return Path(os.getenv("CUSTOM_MODEL_PATH", "DEFAULT_MODEL_PATH"))
    @property
    def IS_SINGLE_MODE(self):
        if self.INPUT_INFERENCE_DATA_PATH and self.INPUT_INFERENCE_DATA_PATH.is_dir():
            return False
        elif self.INPUT_VIDEO_FOLDER_PATH and self.INPUT_VIDEO_FOLDER_PATH.is_dir():
            return False
        return True
    
@dataclass
class PathDict:
    debug_base_path: str = None
    input_video_folder_path: str = None
    input_inference_data_path: str =  None
    log_file_path: str = None
    
    debug_path: str = None
    
    #Derived Paths
    #Note: replace annotation_path to single video_path consisting of all the different annotated video: 9:16 rectabnle, 9:16 person etc...
    stubs_path: str = None #debug_path/stubs
    frames_path: str = None #debug_path/frames
    annotation_path: str = None #debug_path/annotations
    analytics_path: str = None #debug_path/analytics
    camera_view_path: str = None #debug_path/camera_view
    
    #Data paths
    output_video_path: str = None #debug_path/f"output_{debug_path.stem}".mp4
    input_video_path: str = None #debug_path/f"input_{debug_path.stem}".mp4
    input_csv_path: str = None #debug_path/f"input_{debug_path.stem}".csv

@dataclass
class MessageDict:
    game_type: str = None

@dataclass
class StateDict:
    is_debug: str = None
    is_single_mode: str = None
    current_environment: str =None
    outlier_detect: str = None
    is_local: str = None
    outlier_classifier_and_smoother: str = None

@dataclass
class ModelDict:
    model_path: str=  None
    is_custom_model: str= None
    model_type: str=None
    default_model_name: str= None
