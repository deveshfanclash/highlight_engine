"""
Output Module

Typed result classes for inference outputs.
"""

from output.results import (
    BoundingBox,
    Detection,
    Keypoint,
    KeypointSkeleton,
    Results,
    DetectionResults,
    PoseResults,
    ClassificationResults,
)

__all__ = [
    "BoundingBox",
    "Detection",
    "Keypoint",
    "KeypointSkeleton",
    "Results",
    "DetectionResults",
    "PoseResults",
    "ClassificationResults",
]
