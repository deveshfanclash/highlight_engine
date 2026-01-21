import os
from pathlib import Path
from dotenv import load_dotenv
import contextvars
from dataclasses import dataclass
env_path = Path(__file__).parent.parent / '.env' 
print(env_path) #env_path should be told explicitly
load_dotenv(env_path, override=True)
from .config_groups import *
from .constants_groups import *

#----------------------INITIATING GLOBALS-----------------------
path_context = contextvars.ContextVar('path_context', default=None)
message_context = contextvars.ContextVar('message_context', default=None)
model_context = contextvars.ContextVar("model_contex", default=None)
state_context = contextvars.ContextVar("state_context", default=None)
debug_context = contextvars.ContextVar("debug_context", default=None)
#-------------------------User-configurable settings--------------------------

CURRENT_ENVIRONMENT = Environment.PRODUCTION #Change this setting based on requirement.
MODEL_TYPE = ModelType.OBJECT_DETECTION
DEFAULT_MODEL_NAME = ModelName.YOLO_8
IS_CUSTOM_MODEL = False #For class id mapping purpose
IS_DEBUG = True
CUSTOM_MODEL_CLASS_ID_MAPPING = {}
DEFAULT_MODEL_CLASS_ID_MAPPING = {}

#---------------------------------DEBUG CONFIG---------------------------------
@dataclass
class DebugLevels():
    save_annotated_video = True if IS_DEBUG else False
    save_frames = True if IS_DEBUG else False
    log_level = "DEBUG" #DEBUG | INFO | WARNING | ERROR | CRITICAL  
    save_input_data = True if IS_DEBUG else False
    is_video_download = True if IS_DEBUG else False
    is_video_annotate = True if IS_DEBUG else False
    is_full_match_inference = True if IS_DEBUG else False
    tag_instance_wise_inference = False if IS_DEBUG else False
    delete_input_video_downloaded = True if IS_DEBUG else False
    
debug_context.set(DebugLevels())

#---------------------------------ENVIRONMENT---------------------------------
if CURRENT_ENVIRONMENT == Environment.LOCAL:
    env_cfg = LocalConfig()
elif CURRENT_ENVIRONMENT == Environment.PRODUCTION:
    env_cfg = ProdConfig()
elif CURRENT_ENVIRONMENT == Environment.DEVELOPMENT_TESTING:
    env_cfg = DevConfig()
#-----------------------------------Adding context variables ---------------------------------

path_dict = PathDict(
    debug_base_path=Path(env_cfg.DEBUG_BASE_PATH) if hasattr(env_cfg, 'DEBUG_BASE_PATH') else None,
    input_video_folder_path=Path(env_cfg.INPUT_VIDEO_FOLDER_PATH) if hasattr(env_cfg, 'INPUT_VIDEO_FOLDER_PATH') and env_cfg.INPUT_VIDEO_FOLDER_PATH else None,
    input_inference_data_path=Path(env_cfg.INPUT_INFERENCE_DATA_PATH) if hasattr(env_cfg, 'INPUT_INFERENCE_DATA_PATH') and env_cfg.INPUT_INFERENCE_DATA_PATH else None,
    log_file_path=Path(env_cfg.LOG_FILE_PATH) if hasattr(env_cfg, 'LOG_FILE_PATH') else None
)

message_dict = MessageDict(game_type=getattr(env_cfg, 'GAMETYPE', None))

state_dict = StateDict(is_debug=IS_DEBUG,
    is_single_mode=getattr(env_cfg, 'IS_SINGLE_MODE', None),
    current_environment=CURRENT_ENVIRONMENT,
    # outlier_detect=getattr(env_cfg, 'OUTLIER_DETECT', None),
    is_local=getattr(env_cfg, 'IS_LOCAL', None),
    # outlier_classifier_and_smoother=getattr(env_cfg, 'OUTLIER_CLASSIFIER_AND_SMOOTHER', None)
)

model_dict = ModelDict(
    model_path=getattr(env_cfg, 'MODEL_PATH', None),
    is_custom_model=IS_CUSTOM_MODEL,
    model_type=MODEL_TYPE,
    default_model_name=DEFAULT_MODEL_NAME
)

path_context.set(path_dict)
model_context.set(model_dict)
state_context.set(state_dict)
message_context.set(message_dict)
env_cfg.LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
env_cfg.LOG_FILE_PATH.touch(exist_ok=True)

def get_all_contexts():
    sc = state_context.get()
    pc = path_context.get()
    moc = model_context.get()
    mc = message_context.get()
    dc = debug_context.get()
    return sc, pc, moc, mc, dc

# Export environment config for use in other modules
__all__ = ['env_cfg', 'path_context', 'message_context', 'model_context', 
           'state_context', 'debug_context', 'get_all_contexts']
