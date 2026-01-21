"""
Database Module

Provides storage clients for:
- DynamoDB: Frame-level inference results (detections, poses)
- Local files: Testing without AWS
"""

from db.dynamo import (
    DynamoDBWriter,
    DynamoDBWriterConfig,
    LocalFileWriter,
    create_writer,
)

__all__ = [
    "DynamoDBWriter",
    "DynamoDBWriterConfig",
    "LocalFileWriter",
    "create_writer",
]
