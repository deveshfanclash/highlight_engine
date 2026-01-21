import logging
import boto3
import watchtower
import socket
from config.settings import *
from pathlib import Path

try:
    _HOSTNAME = socket.gethostname()
    _MACHINE_IP = socket.gethostbyname(_HOSTNAME)
except Exception:
    _MACHINE_IP = "unknown"

# Initialize the logger early
logger = logging.getLogger(__name__)

try:
    cloudwatch_client = boto3.client(
        "logs",
        aws_access_key_id=env_cfg.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=env_cfg.AWS_SECRET_ACCESS_KEY,
        region_name=env_cfg.AWS_REGION,
    )
    cloudwatch = boto3.client(
        "cloudwatch",
        aws_access_key_id=env_cfg.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=env_cfg.AWS_SECRET_ACCESS_KEY,
        region_name=env_cfg.AWS_REGION,
    )
except Exception as e:
    print(f"Error initializing cloudwatch client: {e}")
    logger.error(f"Error initializing cloudwatch client: {e}")
    cloudwatch_client = None
    cloudwatch = None


logger.setLevel(logging.DEBUG)

class InstanceIDFilter(logging.Filter):
    """Custom filter to replace the logger name with instance ID"""
    def filter(self, record):
        record.machine_ip = _MACHINE_IP
        if 'message_context' in globals():
            msg_ctx= message_context.get()
            if isinstance(msg_ctx, dict):
                metadata= msg_ctx.get('metadata', None)
            else:
                metadata = getattr(msg_ctx, 'metadata', None)
            if metadata:
                tag_instance_id = metadata.get('tag_instance_id', None)
                record.name = tag_instance_id
            else:
                record.name = record.name
        return True

# Add the filter to the logger
instance_filter = InstanceIDFilter()
logger.addFilter(instance_filter)

def configure_logging(
    LOG_FILE_PATH, cloudwatch_enabled=True, cloudwatch_client=None
):
    global logger
    
    # Check if logger already has handlers to avoid duplicate handlers
    if logger.hasHandlers():
        return logger
    
    # File handler for logging to a file
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(machine_ip)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Stream handler for logging to stdout
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter(
        "%(asctime)s - %(machine_ip)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    if cloudwatch_enabled and cloudwatch_client:
        # CloudWatch Handler
        cloudwatch_handler = watchtower.CloudWatchLogHandler(
            log_group=env_cfg.LOG_GROUP_NAME,
            stream_name=str(LOG_FILE_PATH) if isinstance(LOG_FILE_PATH, Path) else LOG_FILE_PATH, #This is a posix path and not a string
            boto3_client=cloudwatch_client,
        )
        cloudwatch_formatter = logging.Formatter(
            "%(name)s - %(machine_ip)s - %(levelname)s - %(message)s"
        )
        cloudwatch_handler.setFormatter(cloudwatch_formatter)
        logger.addHandler(cloudwatch_handler)

    return logger


configure_logging(env_cfg.LOG_FILE_PATH, True, cloudwatch_client)