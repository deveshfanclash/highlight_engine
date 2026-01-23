"""
Database Module

Provides storage clients for:
- DynamoDB: Frame-level inference results (detections, poses)
- Local files: Testing without AWS
"""

from db.dynamo import (
    BaseWriter,
    DynamoDBWriter,
    DynamoDBWriterConfig,
    LocalFileWriter,
    WriterMetrics,
    create_writer,
)

__all__ = [
    "BaseWriter",
    "DynamoDBWriter",
    "DynamoDBWriterConfig",
    "LocalFileWriter",
    "WriterMetrics",
    "create_writer",
]
