"""
Universal Class Registry

Simple mapping between user-friendly class names and model outputs.

Usage:
    # User says they want to detect "ball" and "person"
    classes = ["ball", "person"]

    # System finds matching model class IDs automatically
    model_ids = find_model_classes(model_class_names, classes)
"""

from typing import Dict, List, Optional, Set


# =============================================================================
# UNIVERSAL CLASS DEFINITIONS
# =============================================================================

# Canonical names (uppercase) with their common aliases (lowercase variations)
# Key: canonical name, Value: set of aliases that map to this class
CLASS_ALIASES: Dict[str, Set[str]] = {
    # Common classes
    "PERSON": {"person", "player", "human", "people", "man", "woman"},
    "BALL": {"ball", "sports ball", "sports-ball", "sportsball"},
    "HEAD": {"head", "face"},

    # Football/Soccer
    "GOAL": {"goal", "goalpost", "goal post", "goal-post"},
    "GOALKEEPER": {"goalkeeper", "goalie", "keeper", "gk"},
    "REFEREE": {"referee", "ref", "umpire"},

    # Cricket
    "BAT": {"bat", "cricket bat"},
    "WICKET": {"wicket", "stumps", "stump"},
    "BATSMAN": {"batsman", "batter"},
    "BOWLER": {"bowler"},

    # General sports equipment
    "NET": {"net"},
    "COURT_LINE": {"court line", "line", "boundary line"},
}

# Reverse lookup: alias -> canonical name
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in CLASS_ALIASES.items():
    _ALIAS_TO_CANONICAL[canonical.lower()] = canonical
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def normalize_class_name(name: str) -> str:
    """
    Normalize a class name to its canonical form.

    Args:
        name: Any form of class name ("ball", "BALL", "sports ball", etc.)

    Returns:
        Canonical uppercase name ("BALL")

    Example:
        normalize_class_name("sports ball") -> "BALL"
        normalize_class_name("person") -> "PERSON"
    """
    lower_name = name.lower().strip()
    return _ALIAS_TO_CANONICAL.get(lower_name, name.upper())


def find_model_class_ids(
    model_classes: Dict[int, str],
    requested_classes: List[str]
) -> List[int]:
    """
    Find model's native class IDs for user-requested class names.

    This is the main function - it bridges user input to model output.

    Args:
        model_classes: Model's native class mapping {id: name}
                      e.g., {0: "person", 1: "bicycle", 32: "sports ball"}
        requested_classes: User's requested classes ["person", "ball"]

    Returns:
        List of model class IDs to use for inference
        e.g., [0, 32]

    Example:
        model_classes = {0: "person", 32: "sports ball", 37: "tennis racket"}
        requested = ["ball", "person"]
        find_model_class_ids(model_classes, requested) -> [0, 32]
    """
    if not requested_classes:
        return []  # Empty = predict all classes

    # Normalize requested class names
    normalized_requested = {normalize_class_name(c) for c in requested_classes}

    # Build reverse lookup for model classes
    # model_name -> model_id, normalized to canonical names
    model_name_to_id: Dict[str, int] = {}
    for model_id, model_name in model_classes.items():
        canonical = normalize_class_name(model_name)
        model_name_to_id[canonical] = model_id
        # Also store the lowercase original for direct matching
        model_name_to_id[model_name.lower()] = model_id

    # Find matching IDs
    found_ids = []
    for requested in normalized_requested:
        if requested in model_name_to_id:
            found_ids.append(model_name_to_id[requested])
        elif requested.lower() in model_name_to_id:
            found_ids.append(model_name_to_id[requested.lower()])

    return sorted(found_ids)


def create_class_mapping(
    model_classes: Dict[int, str]
) -> Dict[int, str]:
    """
    Create a mapping from model class IDs to canonical names.

    Args:
        model_classes: Model's native class mapping {id: name}

    Returns:
        Mapping {model_id: canonical_name}

    Example:
        model_classes = {0: "person", 32: "sports ball"}
        create_class_mapping(model_classes) -> {0: "PERSON", 32: "BALL"}
    """
    return {
        model_id: normalize_class_name(model_name)
        for model_id, model_name in model_classes.items()
    }


def get_canonical_names() -> List[str]:
    """Get all canonical class names."""
    return list(CLASS_ALIASES.keys())


def add_alias(canonical: str, alias: str):
    """
    Add a new alias for a canonical class name.

    Useful for custom models with non-standard class names.

    Args:
        canonical: Canonical name (e.g., "BALL")
        alias: New alias (e.g., "soccer ball")
    """
    canonical_upper = canonical.upper()
    if canonical_upper not in CLASS_ALIASES:
        CLASS_ALIASES[canonical_upper] = set()
    CLASS_ALIASES[canonical_upper].add(alias.lower())
    _ALIAS_TO_CANONICAL[alias.lower()] = canonical_upper


# =============================================================================
# COCO CLASS NAMES (Common baseline)
# =============================================================================

# COCO dataset class names (used by most pretrained YOLO models)
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}


if __name__ == "__main__":
    # Test the functions
    print("Testing class registry...")

    # Test normalization
    assert normalize_class_name("ball") == "BALL"
    assert normalize_class_name("sports ball") == "BALL"
    assert normalize_class_name("PERSON") == "PERSON"
    assert normalize_class_name("player") == "PERSON"
    print("✓ Normalization works")

    # Test finding model class IDs
    model = {0: "person", 32: "sports ball", 37: "tennis racket"}
    ids = find_model_class_ids(model, ["ball", "person"])
    assert set(ids) == {0, 32}
    print("✓ Finding model class IDs works")

    # Test with COCO classes
    ids = find_model_class_ids(COCO_CLASSES, ["ball", "person"])
    assert set(ids) == {0, 32}
    print("✓ COCO class lookup works")

    print("\nAll tests passed!")
