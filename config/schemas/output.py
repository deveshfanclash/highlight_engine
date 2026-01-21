"""
Output Schemas

Schemas for what gets written to DynamoDB.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box (0-1 range)"""
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)


class Detection(BaseModel):
    """Single detection result"""
    class_name: str = Field(..., description="Universal class name")
    class_id: int = Field(..., description="Universal class ID")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox


class InferenceOutput(BaseModel):
    """
    Output schema for inference results.
    Written to DynamoDB table: inference_results

    PK: {match_id}#{service_id}
    SK: {frame_number}
    """
    # Keys (for DynamoDB)
    pk: str = Field(..., description="Partition key: match_id#service_id")
    sk: int = Field(..., description="Sort key: frame_number")

    # Identifiers
    match_id: str
    service_id: str = Field(..., description="Service identifier (e.g., 'od_football_v2')")
    model_id: str = Field(..., description="Model that produced these detections")
    frame_number: int
    timestamp_ms: int = Field(..., description="Frame timestamp in video (milliseconds)")

    # Detections
    detections: List[Detection] = Field(default_factory=list)

    # Metadata
    processing_time_ms: int = Field(..., description="Time to process this frame")
    written_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def create_pk(cls, match_id: str, service_id: str) -> str:
        return f"{match_id}#{service_id}"


class Keypoint(BaseModel):
    """Single keypoint in pose estimation"""
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized x coordinate")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized y coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0)
    name: Optional[str] = Field(None, description="Keypoint name (e.g., 'left_shoulder')")


class PoseDetection(BaseModel):
    """Single pose detection result"""
    person_id: Optional[int] = Field(None, description="Tracked person ID if available")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box of the person")
    keypoints: List[Keypoint] = Field(default_factory=list)


class PoseOutput(BaseModel):
    """
    Output schema for pose estimation results.
    Written to DynamoDB table: inference_results (same table, different service_id)

    PK: {match_id}#{service_id}
    SK: {frame_number}
    """
    pk: str = Field(..., description="Partition key: match_id#service_id")
    sk: int = Field(..., description="Sort key: frame_number")

    match_id: str
    service_id: str = Field(..., description="Service identifier")
    model_id: str = Field(..., description="Model that produced these detections")
    frame_number: int
    timestamp_ms: int = Field(..., description="Frame timestamp in video (milliseconds)")

    # Pose detections
    poses: List[PoseDetection] = Field(default_factory=list)

    # Metadata
    processing_time_ms: int = Field(..., description="Time to process this frame")
    written_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def create_pk(cls, match_id: str, service_id: str) -> str:
        return f"{match_id}#{service_id}"
