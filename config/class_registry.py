"""
Universal Class Registry

Defines all universal classes used across sports/games.
Models map their native class IDs to these universal class names.

This serves as the single source of truth for class definitions.
Changes here affect how detections are labeled across all games.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import IntEnum


class UniversalClassID(IntEnum):
    """
    Universal Class IDs - consistent across all models and games.

    Convention:
    - 0: Reserved (unused)
    - 1-99: Common objects (person, ball, etc.)
    - 100-199: Football-specific
    - 200-299: Cricket-specific
    - 300-399: Volleyball-specific
    - 400-499: Tennis/Racket sports
    - 500-599: Combat sports
    - 900-999: Miscellaneous/Other
    """
    # Common (1-99)
    PERSON = 1
    BALL = 2
    HEAD = 3
    REFEREE = 4
    GOALKEEPER = 5

    # Football-specific (100-199)
    GOAL = 100
    GOAL_POST = 101
    CORNER_FLAG = 102
    FOOTBALL = 103  # Specific ball type

    # Cricket-specific (200-299)
    BATSMAN = 200
    BOWLER = 201
    WICKET_KEEPER = 202
    FIELDER = 203
    WICKET = 204
    STUMP = 205
    CRICKET_BALL = 206
    BAT = 207
    CREASE = 208
    BOUNDARY = 209

    # Volleyball-specific (300-399)
    NET = 300
    VOLLEYBALL = 301
    COURT_LINE = 302

    # Tennis/Racket (400-499)
    TENNIS_BALL = 400
    TENNIS_RACKET = 401
    TENNIS_NET = 402
    BADMINTON_SHUTTLECOCK = 410
    TABLE_TENNIS_BALL = 420
    TABLE_TENNIS_PADDLE = 421
    TABLE_TENNIS_TABLE = 422

    # Combat sports (500-599)
    BOXER = 500
    BOXING_RING = 501
    BOXING_GLOVE = 502

    # Miscellaneous (900-999)
    SCOREBOARD = 900
    CAMERA = 901
    ADVERTISEMENT = 902
    CROWD = 903


@dataclass
class ClassDefinition:
    """Complete definition of a universal class"""
    class_id: int
    class_name: str
    description: str
    applicable_sports: List[str]  # List of game names, or ["all"]
    parent_class: Optional[str] = None  # For hierarchical classes (e.g., BATSMAN is-a PERSON)


# Master registry of all universal classes
UNIVERSAL_CLASS_REGISTRY: Dict[str, ClassDefinition] = {
    # ==========================================================================
    # COMMON CLASSES (apply to all/most sports)
    # ==========================================================================
    "PERSON": ClassDefinition(
        class_id=UniversalClassID.PERSON,
        class_name="PERSON",
        description="Generic human/player on field",
        applicable_sports=["all"],
    ),
    "BALL": ClassDefinition(
        class_id=UniversalClassID.BALL,
        class_name="BALL",
        description="Generic ball (sport-agnostic)",
        applicable_sports=["all"],
    ),
    "HEAD": ClassDefinition(
        class_id=UniversalClassID.HEAD,
        class_name="HEAD",
        description="Human head detection",
        applicable_sports=["all"],
    ),
    "REFEREE": ClassDefinition(
        class_id=UniversalClassID.REFEREE,
        class_name="REFEREE",
        description="Match referee/umpire",
        applicable_sports=["all"],
    ),
    "GOALKEEPER": ClassDefinition(
        class_id=UniversalClassID.GOALKEEPER,
        class_name="GOALKEEPER",
        description="Goalkeeper/keeper in goal-based sports",
        applicable_sports=["football", "hockey", "handball"],
    ),

    # ==========================================================================
    # FOOTBALL CLASSES
    # ==========================================================================
    "GOAL": ClassDefinition(
        class_id=UniversalClassID.GOAL,
        class_name="GOAL",
        description="Goal structure/net",
        applicable_sports=["football", "hockey", "handball"],
    ),
    "GOAL_POST": ClassDefinition(
        class_id=UniversalClassID.GOAL_POST,
        class_name="GOAL_POST",
        description="Goal post (vertical bar)",
        applicable_sports=["football"],
    ),
    "CORNER_FLAG": ClassDefinition(
        class_id=UniversalClassID.CORNER_FLAG,
        class_name="CORNER_FLAG",
        description="Corner flag marker",
        applicable_sports=["football"],
    ),
    "FOOTBALL": ClassDefinition(
        class_id=UniversalClassID.FOOTBALL,
        class_name="FOOTBALL",
        description="Football/soccer ball specifically",
        applicable_sports=["football"],
        parent_class="BALL",
    ),

    # ==========================================================================
    # CRICKET CLASSES
    # ==========================================================================
    "BATSMAN": ClassDefinition(
        class_id=UniversalClassID.BATSMAN,
        class_name="BATSMAN",
        description="Cricket batsman",
        applicable_sports=["cricket"],
        parent_class="PERSON",
    ),
    "BOWLER": ClassDefinition(
        class_id=UniversalClassID.BOWLER,
        class_name="BOWLER",
        description="Cricket bowler",
        applicable_sports=["cricket"],
        parent_class="PERSON",
    ),
    "WICKET_KEEPER": ClassDefinition(
        class_id=UniversalClassID.WICKET_KEEPER,
        class_name="WICKET_KEEPER",
        description="Cricket wicket keeper",
        applicable_sports=["cricket"],
        parent_class="PERSON",
    ),
    "FIELDER": ClassDefinition(
        class_id=UniversalClassID.FIELDER,
        class_name="FIELDER",
        description="Cricket fielder",
        applicable_sports=["cricket"],
        parent_class="PERSON",
    ),
    "WICKET": ClassDefinition(
        class_id=UniversalClassID.WICKET,
        class_name="WICKET",
        description="Cricket wicket (3 stumps + bails)",
        applicable_sports=["cricket"],
    ),
    "STUMP": ClassDefinition(
        class_id=UniversalClassID.STUMP,
        class_name="STUMP",
        description="Individual cricket stump",
        applicable_sports=["cricket"],
    ),
    "CRICKET_BALL": ClassDefinition(
        class_id=UniversalClassID.CRICKET_BALL,
        class_name="CRICKET_BALL",
        description="Cricket ball",
        applicable_sports=["cricket"],
        parent_class="BALL",
    ),
    "BAT": ClassDefinition(
        class_id=UniversalClassID.BAT,
        class_name="BAT",
        description="Cricket bat",
        applicable_sports=["cricket"],
    ),
    "CREASE": ClassDefinition(
        class_id=UniversalClassID.CREASE,
        class_name="CREASE",
        description="Cricket crease line",
        applicable_sports=["cricket"],
    ),
    "BOUNDARY": ClassDefinition(
        class_id=UniversalClassID.BOUNDARY,
        class_name="BOUNDARY",
        description="Cricket boundary rope/line",
        applicable_sports=["cricket"],
    ),

    # ==========================================================================
    # VOLLEYBALL CLASSES
    # ==========================================================================
    "NET": ClassDefinition(
        class_id=UniversalClassID.NET,
        class_name="NET",
        description="Volleyball/tennis net",
        applicable_sports=["volleyball", "tennis", "badminton"],
    ),
    "VOLLEYBALL": ClassDefinition(
        class_id=UniversalClassID.VOLLEYBALL,
        class_name="VOLLEYBALL",
        description="Volleyball specifically",
        applicable_sports=["volleyball"],
        parent_class="BALL",
    ),
    "COURT_LINE": ClassDefinition(
        class_id=UniversalClassID.COURT_LINE,
        class_name="COURT_LINE",
        description="Court boundary line",
        applicable_sports=["volleyball", "tennis", "basketball"],
    ),

    # ==========================================================================
    # TENNIS / RACKET SPORT CLASSES
    # ==========================================================================
    "TENNIS_BALL": ClassDefinition(
        class_id=UniversalClassID.TENNIS_BALL,
        class_name="TENNIS_BALL",
        description="Tennis ball",
        applicable_sports=["tennis"],
        parent_class="BALL",
    ),
    "TENNIS_RACKET": ClassDefinition(
        class_id=UniversalClassID.TENNIS_RACKET,
        class_name="TENNIS_RACKET",
        description="Tennis racket",
        applicable_sports=["tennis"],
    ),
    "BADMINTON_SHUTTLECOCK": ClassDefinition(
        class_id=UniversalClassID.BADMINTON_SHUTTLECOCK,
        class_name="BADMINTON_SHUTTLECOCK",
        description="Badminton shuttlecock",
        applicable_sports=["badminton"],
    ),
    "TABLE_TENNIS_BALL": ClassDefinition(
        class_id=UniversalClassID.TABLE_TENNIS_BALL,
        class_name="TABLE_TENNIS_BALL",
        description="Table tennis ball",
        applicable_sports=["table_tennis"],
        parent_class="BALL",
    ),
    "TABLE_TENNIS_PADDLE": ClassDefinition(
        class_id=UniversalClassID.TABLE_TENNIS_PADDLE,
        class_name="TABLE_TENNIS_PADDLE",
        description="Table tennis paddle/bat",
        applicable_sports=["table_tennis"],
    ),
    "TABLE_TENNIS_TABLE": ClassDefinition(
        class_id=UniversalClassID.TABLE_TENNIS_TABLE,
        class_name="TABLE_TENNIS_TABLE",
        description="Table tennis table",
        applicable_sports=["table_tennis"],
    ),

    # ==========================================================================
    # MISCELLANEOUS
    # ==========================================================================
    "SCOREBOARD": ClassDefinition(
        class_id=UniversalClassID.SCOREBOARD,
        class_name="SCOREBOARD",
        description="Scoreboard/score display",
        applicable_sports=["all"],
    ),
    "CROWD": ClassDefinition(
        class_id=UniversalClassID.CROWD,
        class_name="CROWD",
        description="Crowd/spectators",
        applicable_sports=["all"],
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_class_by_name(class_name: str) -> Optional[ClassDefinition]:
    """Get class definition by name"""
    return UNIVERSAL_CLASS_REGISTRY.get(class_name.upper())


def get_class_by_id(class_id: int) -> Optional[ClassDefinition]:
    """Get class definition by ID"""
    for class_def in UNIVERSAL_CLASS_REGISTRY.values():
        if class_def.class_id == class_id:
            return class_def
    return None


def get_classes_for_sport(sport_name: str) -> List[ClassDefinition]:
    """Get all classes applicable to a specific sport"""
    result = []
    sport_lower = sport_name.lower()
    for class_def in UNIVERSAL_CLASS_REGISTRY.values():
        if "all" in class_def.applicable_sports or sport_lower in class_def.applicable_sports:
            result.append(class_def)
    return result


def get_class_id(class_name: str) -> Optional[int]:
    """Get class ID by name (convenience function)"""
    class_def = get_class_by_name(class_name)
    return class_def.class_id if class_def else None


def get_class_name(class_id: int) -> Optional[str]:
    """Get class name by ID (convenience function)"""
    class_def = get_class_by_id(class_id)
    return class_def.class_name if class_def else None


def validate_class_name(class_name: str) -> bool:
    """Check if class name is valid"""
    return class_name.upper() in UNIVERSAL_CLASS_REGISTRY


def get_all_class_names() -> List[str]:
    """Get all registered class names"""
    return list(UNIVERSAL_CLASS_REGISTRY.keys())


# =============================================================================
# MAPPING HELPERS (for model output conversion)
# =============================================================================

def create_model_to_universal_mapping(
    model_classes: Dict[int, str]
) -> Dict[int, int]:
    """
    Create a mapping from model class IDs to universal class IDs.

    Args:
        model_classes: Dict mapping model class ID to universal class name
                      e.g., {0: "PERSON", 1: "BALL", 32: "GOAL"}

    Returns:
        Dict mapping model class ID to universal class ID
        e.g., {0: 1, 1: 2, 32: 100}
    """
    mapping = {}
    for model_id, class_name in model_classes.items():
        class_def = get_class_by_name(class_name)
        if class_def:
            mapping[model_id] = class_def.class_id
        else:
            raise ValueError(f"Unknown class name: {class_name}")
    return mapping


# =============================================================================
# QUICK REFERENCE (for documentation)
# =============================================================================

def print_registry_summary():
    """Print a summary of all registered classes (for documentation)"""
    print("=" * 60)
    print("UNIVERSAL CLASS REGISTRY")
    print("=" * 60)

    # Group by sport
    sports = set()
    for class_def in UNIVERSAL_CLASS_REGISTRY.values():
        for sport in class_def.applicable_sports:
            sports.add(sport)

    for sport in sorted(sports):
        print(f"\n{sport.upper()}:")
        print("-" * 40)
        classes = get_classes_for_sport(sport) if sport != "all" else [
            c for c in UNIVERSAL_CLASS_REGISTRY.values() if "all" in c.applicable_sports
        ]
        for c in sorted(classes, key=lambda x: x.class_id):
            print(f"  {c.class_id:4d} | {c.class_name:25s} | {c.description}")


if __name__ == "__main__":
    print_registry_summary()
