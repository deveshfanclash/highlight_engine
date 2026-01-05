"""
Database Module

Provides database clients for:
- DynamoDB (inference results)
- MongoDB (configuration)
"""

from db.dynamo import DynamoDBWriter, DynamoDBWriterConfig

__all__ = [
    "DynamoDBWriter",
    "DynamoDBWriterConfig",
]
