"""
Match Configuration (Tier 3)

Runtime configuration for a specific match.
This is what gets created when a match starts.

MatchConfig:
- References GameTemplate (Tier 2) by game_id
- Contains stream URL and match metadata
- Supports overrides for this specific match
- Tracks runtime state (status, resume position)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from config.schemas.enums import MatchStatus, InputType
from config.schemas.model import ModelParams
from config.schemas.game import InferenceOverrides, OutputOverrides


# =============================================================================
# RESUME POSITION
# =============================================================================

class ResumePosition(BaseModel):
    """
    Tracks where to resume processing from.

    Used for crash recovery and match continuation.
    """
    segment_number: int = Field(default=1, description="HLS segment number")
    frame_number: int = Field(default=0, description="Frame number within stream")
    timestamp_ms: int = Field(default=0, description="Timestamp in milliseconds")
    last_updated: Optional[datetime] = None


# =============================================================================
# MATCH OVERRIDES
# =============================================================================

class ServiceOverride(BaseModel):
    """Override settings for a specific service in this match"""
    enabled: Optional[bool] = None
    frame_skip: Optional[int] = None

    # Service-specific overrides (generic dict)
    params: Dict[str, Any] = Field(default_factory=dict)


class MatchOverrides(BaseModel):
    """
    Overrides for this specific match.

    These take precedence over GameTemplate and ModelConfig defaults.

    Override resolution order:
    1. Game defaults (GameTemplate.defaults)
    2. Service overrides (service.inference_overrides)
    3. Match overrides (MatchConfig.overrides) <- Highest priority
    """

    # Inference settings override (partial - only specify what differs)
    inference_overrides: Optional[InferenceOverrides] = Field(
        None,
        description="Override inference settings for this match"
    )

    # Output settings override (partial - only specify what differs)
    output_overrides: Optional[OutputOverrides] = Field(
        None,
        description="Override output settings for this match"
    )

    # Model parameter overrides (model_id -> params)
    model_params: Dict[str, ModelParams] = Field(
        default_factory=dict,
        description="Override model params by model_id"
    )

    # Service overrides (service_type -> settings)
    service_overrides: Dict[str, ServiceOverride] = Field(
        default_factory=dict,
        description="Override service settings by service_type"
    )

    # Class mapping overrides
    class_mapping_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description="Override specific class mappings"
    )

    class Config:
        protected_namespaces = ()


# =============================================================================
# MATCH METADATA
# =============================================================================

class MatchMetadata(BaseModel):
    """Metadata about the match (for logging/analytics)"""
    league: Optional[str] = None
    tournament_id: Optional[str] = None

    # Custom fields
    extra: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# MATCH CONFIGURATION
# =============================================================================

class MatchConfig(BaseModel):
    """
    Match Configuration (Tier 3).

    Runtime configuration for a specific match.

    References:
    - GameTemplate (Tier 2) via game_id

    Contains:
    - Stream URL and type
    - Match-specific overrides
    - Runtime state (status, resume position)
    - Metadata for logging
    """

    # -------------------------------------------------------------------------
    # IDENTITY
    # -------------------------------------------------------------------------
    match_id: str = Field(..., description="Unique match identifier")

    # -------------------------------------------------------------------------
    # REFERENCES (Links to other tiers)
    # -------------------------------------------------------------------------
    game_id: str = Field(..., description="Reference to GameTemplate (Tier 2)")

    # -------------------------------------------------------------------------
    # INPUT CONFIGURATION
    # -------------------------------------------------------------------------
    stream_url: str = Field(..., description="Input stream URL (HLS, RTSP, etc.)")
    stream_type: InputType = Field(default=InputType.HLS)

    # -------------------------------------------------------------------------
    # OVERRIDES
    # -------------------------------------------------------------------------
    overrides: Optional[MatchOverrides] = Field(
        None,
        description="Match-specific overrides (highest priority)"
    )

    # -------------------------------------------------------------------------
    # RUNTIME STATE
    # -------------------------------------------------------------------------
    status: MatchStatus = Field(default=MatchStatus.PENDING)
    resume_position: ResumePosition = Field(default_factory=ResumePosition)

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------
    metadata: MatchMetadata = Field(default_factory=MatchMetadata)

    class Config:
        use_enum_values = True

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def get_model_params_override(self, model_id: str) -> Optional[ModelParams]:
        """Get model params override for a specific model"""
        if self.overrides and model_id in self.overrides.model_params:
            return self.overrides.model_params[model_id]
        return None

    def get_service_override(self, service_type: str) -> Optional[ServiceOverride]:
        """Get service override for a specific service type"""
        if self.overrides and service_type in self.overrides.service_overrides:
            return self.overrides.service_overrides[service_type]
        return None

    def mark_started(self):
        """Mark match as started"""
        self.status = MatchStatus.RUNNING
        self.started_at = datetime.utcnow()

    def mark_completed(self):
        """Mark match as completed"""
        self.status = MatchStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error: str):
        """Mark match as failed"""
        self.status = MatchStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error

    def update_resume_position(self, segment: int, frame: int, timestamp_ms: int = 0):
        """Update resume position"""
        self.resume_position.segment_number = segment
        self.resume_position.frame_number = frame
        self.resume_position.timestamp_ms = timestamp_ms
        self.resume_position.last_updated = datetime.utcnow()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_match_config(
    match_id: str,
    stream_url: str,
    game_id: str,
    stream_type: InputType = InputType.HLS,
    overrides: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MatchConfig:
    """
    Factory function to create a match config.

    Args:
        match_id: Unique match identifier
        stream_url: HLS/RTSP stream URL
        game_id: Reference to GameTemplate
        stream_type: Type of stream
        overrides: Optional override dict
        metadata: Optional metadata dict

    Returns:
        MatchConfig instance
    """
    match_overrides = None
    if overrides:
        match_overrides = MatchOverrides(**overrides)

    match_metadata = MatchMetadata()
    if metadata:
        match_metadata = MatchMetadata(**metadata)

    return MatchConfig(
        match_id=match_id,
        stream_url=stream_url,
        game_id=game_id,
        stream_type=stream_type,
        overrides=match_overrides,
        metadata=match_metadata,
    )
