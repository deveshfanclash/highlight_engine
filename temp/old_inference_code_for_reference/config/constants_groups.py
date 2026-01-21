from enum import Enum
from dataclasses import dataclass

class MODEL_CONSTANTS(Enum):
    YOLO_BATCH_SIZE = 20


class CRICKET_CONSTANTS(Enum):
    avg_person_area_threshold=50000,
    cricket_accuracy_threshold=0.5,
    cricket_accuracy_ratio=0.2,
    cricket_smoothing_frames=10,
    head_width_threshold=609,
    person_width_margin=305,
    person_center_offset=250,
    smoothing_window_size=5,
    outlier_threshold=1.5,
    velocity_threshold=30,
    proximity_threshold=150