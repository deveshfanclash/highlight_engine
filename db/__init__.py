"""
Database & Storage Module

Provides storage clients for:
- DynamoDB: Frame-level inference results (detections, camera cuts)
- S3: Large outputs (segmentation masks, keypoints, embeddings)
- MongoDB: Configuration (via config.loader)
- Local files: Testing without AWS
"""

from db.dynamo import (
    DynamoDBWriter,
    DynamoDBWriterConfig,
    LocalFileWriter,
    create_writer,
)
from db.s3_storage import S3Storage, S3StorageConfig, create_storage

__all__ = [
    # DynamoDB
    "DynamoDBWriter",
    "DynamoDBWriterConfig",
    "LocalFileWriter",
    "create_writer",
    # S3 Storage
    "S3Storage",
    "S3StorageConfig",
    "create_storage",
]
