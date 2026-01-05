"""
Output Schemas for Inference Services

This module defines the output data schemas for all inference services.
These schemas document the data format written to DynamoDB and local files,
enabling sync with downstream post-processing repositories.

DynamoDB Table Structure:
- Single table design with composite keys
- PK (Partition Key): "{match_id}#{service_id}" for most services
- SK (Sort Key): frame_number (int) for most services

Services:
- Object Detection (OD): One row per model per frame
- Camera View: One row per detected camera cut
- HLS Metadata: One row per frame (for resume functionality)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class ServiceType(str, Enum):
    """Types of inference services"""
    OBJECT_DETECTION = "object_detection"
    CAMERA_VIEW = "camera_view"
    HLS_METADATA = "hls_metadata"


# =============================================================================
# DETECTION SCHEMA
# =============================================================================

class Detection(BaseModel):
    """
    Single object detection result.

    Bounding box coordinates are NORMALIZED (0-1 range) for resolution independence.
    To convert to pixel coordinates: x_pixel = x_norm * image_width
    """
    class_id: int = Field(..., description="Universal class ID from class registry")
    class_name: str = Field(..., description="Universal class name (e.g., 'PERSON', 'BALL')")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence score (0-1)")
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] normalized to 0-1 range"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "class_id": 1,
                "class_name": "PERSON",
                "confidence": 0.95,
                "bbox": [0.123, 0.456, 0.234, 0.567]
            }
        }


# =============================================================================
# OBJECT DETECTION OUTPUT
# =============================================================================

class ObjectDetectionOutput(BaseModel):
    """
    Output schema for Object Detection service.

    DynamoDB Keys:
    - PK: "{match_id}#{service_id}" (e.g., "match_123#od_football_v2")
    - SK: frame_number

    Query Patterns:
    - All OD results for match: pk begins_with "match_123#od_"
    - Specific model results: pk = "match_123#od_football_v2"
    - Frame range: pk = "..." AND sk BETWEEN 1000 AND 2000
    """
    # DynamoDB Keys
    pk: str = Field(..., description="Partition key: {match_id}#{service_id}")
    sk: int = Field(..., description="Sort key: frame_number")

    # Identifiers
    match_id: str = Field(..., description="Match identifier")
    service_id: str = Field(..., description="Service identifier (e.g., 'od_football_v2')")
    model_id: str = Field(..., description="Model identifier that produced detections")

    # Frame metadata
    frame_number: int = Field(..., ge=0, description="Frame number (0-indexed)")
    timestamp_ms: int = Field(..., ge=0, description="Frame timestamp in milliseconds")
    processing_time_ms: int = Field(..., ge=0, description="Inference processing time in ms")

    # Detection results
    detections: List[Detection] = Field(
        default_factory=list,
        description="List of detections for this frame"
    )

    # Optional metadata
    written_at: Optional[str] = Field(None, description="ISO timestamp when written to DB")

    class Config:
        json_schema_extra = {
            "example": {
                "pk": "match_123#od_football_v2",
                "sk": 1500,
                "match_id": "match_123",
                "service_id": "od_football_v2",
                "model_id": "football_v2",
                "frame_number": 1500,
                "timestamp_ms": 60000,
                "processing_time_ms": 45,
                "detections": [
                    {
                        "class_id": 1,
                        "class_name": "PERSON",
                        "confidence": 0.95,
                        "bbox": [0.1, 0.2, 0.15, 0.4]
                    },
                    {
                        "class_id": 2,
                        "class_name": "BALL",
                        "confidence": 0.87,
                        "bbox": [0.5, 0.6, 0.52, 0.62]
                    }
                ],
                "written_at": "2024-01-05T12:30:45.123Z"
            }
        }


# =============================================================================
# CAMERA VIEW OUTPUT
# =============================================================================

class CameraViewOutput(BaseModel):
    """
    Output schema for Camera View Detection service.

    Only writes a record when a camera cut is detected.

    DynamoDB Keys:
    - PK: "{match_id}#camera_view"
    - SK: frame_number

    Query Patterns:
    - All camera cuts for match: pk = "match_123#camera_view"
    - Cuts in time range: pk = "..." AND sk BETWEEN 1000 AND 5000
    """
    # DynamoDB Keys
    pk: str = Field(..., description="Partition key: {match_id}#camera_view")
    sk: int = Field(..., description="Sort key: frame_number")

    # Identifiers
    match_id: str = Field(..., description="Match identifier")
    service_id: str = Field(default="camera_view", description="Always 'camera_view'")

    # Frame metadata
    frame_number: int = Field(..., ge=0, description="Frame number where cut detected")
    timestamp_ms: int = Field(..., ge=0, description="Frame timestamp in milliseconds")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in ms")

    # Detection results
    is_camera_cut: bool = Field(default=True, description="Always True (only writes on cut)")
    phash_diff: Optional[int] = Field(
        None,
        description="Perceptual hash difference (Hamming distance)"
    )
    histogram_correlation: Optional[float] = Field(
        None,
        ge=-1,
        le=1,
        description="Histogram correlation coefficient (-1 to 1)"
    )

    # Optional metadata
    written_at: Optional[str] = Field(None, description="ISO timestamp when written to DB")

    class Config:
        json_schema_extra = {
            "example": {
                "pk": "match_123#camera_view",
                "sk": 2500,
                "match_id": "match_123",
                "service_id": "camera_view",
                "frame_number": 2500,
                "timestamp_ms": 100000,
                "processing_time_ms": 12,
                "is_camera_cut": True,
                "phash_diff": 24,
                "histogram_correlation": 0.45,
                "written_at": "2024-01-05T12:30:45.123Z"
            }
        }


# =============================================================================
# HLS METADATA OUTPUT
# =============================================================================

class HLSMetadataOutput(BaseModel):
    """
    Output schema for HLS Stream Metadata service.

    Maps frames to HLS segments for resume functionality.

    DynamoDB Keys:
    - PK: match_id
    - SK: ptstime (float) - Presentation Timestamp

    Query Patterns:
    - All metadata for match: pk = "match_123"
    - Latest N frames: pk = "match_123", ScanIndexForward=False, Limit=N
    - Resume query: Get latest segment_number and frame_number
    """
    # DynamoDB Keys
    pk: str = Field(..., description="Partition key: match_id")
    sk: float = Field(..., description="Sort key: ptstime (PTS in seconds)")

    # Identifiers
    match_id: str = Field(..., description="Match identifier")

    # Segment information
    segment: str = Field(..., description="Segment filename (e.g., 'segment_00123.ts')")
    segment_number: int = Field(..., ge=0, description="Segment number (extracted from filename)")

    # Frame information
    frame_number: int = Field(..., ge=1, description="Global frame number (1-indexed)")
    ts_frame: int = Field(..., ge=0, description="Frame index within the segment (0-indexed)")
    ptstime: float = Field(..., ge=0, description="Presentation timestamp in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "pk": "match_123",
                "sk": 125.567,
                "match_id": "match_123",
                "segment": "segments_480p_20240105T123045_00015.ts",
                "segment_number": 15,
                "frame_number": 3142,
                "ts_frame": 42,
                "ptstime": 125.567
            }
        }


# =============================================================================
# LEGACY SCHEMA COMPATIBILITY (Old Code Format)
# =============================================================================

# class LegacyODOutput(BaseModel):
#     """
#     Legacy output format for backward compatibility with old post-processing.

#     WARNING: Old code had a bug where class_id was overwritten with timestamp!
#     This schema documents the INTENDED format (with actual class_ids).

#     DynamoDB Keys (Old):
#     - PK: match_id
#     - SK: frame_number
#     """
#     match_id: str
#     frame_number: int
#     bbox: List[List[float]] = Field(..., description="[[x1,y1,x2,y2], ...] in PIXEL coordinates")
#     confidence: List[float]
#     class_id: List[int] = Field(..., description="List of class IDs (was bugged in old code)")

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "match_id": "match_123",
#                 "frame_number": 1500,
#                 "bbox": [[100.5, 200.3, 150.2, 400.8], [500.0, 300.0, 520.0, 340.0]],
#                 "confidence": [0.95, 0.87],
#                 "class_id": [0, 1]
#             }
#         }


# class LegacyCricketODOutput(BaseModel):
#     """
#     Legacy Cricket output format (3 models merged into single row).

#     DynamoDB Keys (Old):
#     - PK: match_id
#     - SK: frame_number

#     NOTE: New code writes 3 separate rows (one per model) instead.
#     """
#     match_id: str
#     frame_number: int

#     # Cricket model detections
#     cricket_model_bbox: List[List[float]]
#     cricket_model_confidence: List[float]
#     cricket_model_class_id: List[int]

#     # Person head model detections
#     person_head_model_bbox: List[List[float]]
#     person_head_model_confidence: List[float]
#     person_head_model_class_id: List[int]

#     # Person detection model detections
#     person_detection_model_bbox: List[List[float]]
#     person_detection_model_confidence: List[float]
#     person_detection_model_class_id: List[int]


# class LegacyCameraChangeOutput(BaseModel):
#     """
#     Legacy Camera Change output format.

#     DynamoDB Keys (Old):
#     - PK: match_id
#     - SK: frame_number
#     """
#     match_id: str
#     frame_number: int


# =============================================================================
# SCHEMA MIGRATION HELPERS
# =============================================================================

# def convert_new_to_legacy_od(new_output: ObjectDetectionOutput, image_width: int, image_height: int) -> LegacyODOutput:
#     """
#     Convert new OD output format to legacy format.

#     Args:
#         new_output: New format output
#         image_width: Original image width for denormalization
#         image_height: Original image height for denormalization

#     Returns:
#         Legacy format output
#     """
#     bboxes = []
#     confidences = []
#     class_ids = []

#     for det in new_output.detections:
#         # Denormalize bbox
#         x1 = det.bbox[0] * image_width
#         y1 = det.bbox[1] * image_height
#         x2 = det.bbox[2] * image_width
#         y2 = det.bbox[3] * image_height
#         bboxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
#         confidences.append(round(det.confidence, 2))
#         class_ids.append(det.class_id)

#     return LegacyODOutput(
#         match_id=new_output.match_id,
#         frame_number=new_output.frame_number,
#         bbox=bboxes,
#         confidence=confidences,
#         class_id=class_ids
#     )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ServiceType",

    # Core schemas
    "Detection",
    "ObjectDetectionOutput",
    "CameraViewOutput",
    "HLSMetadataOutput",

    # Legacy schemas
    "LegacyODOutput",
    "LegacyCricketODOutput",
    "LegacyCameraChangeOutput",

    # Helpers
    "convert_new_to_legacy_od",
]
