"""
Deployment Profile Configuration (Tier 3)

Defines WHERE and HOW to run services:
- Development: Local file output, no AWS
- Production: DynamoDB, AWS Batch, etc.

This separates infrastructure concerns from game/model logic.
"""

from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class Environment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"

class InstanceType(str, Enum):
    """Common AWS instance types for inference"""
    # CPU instances
    C5_LARGE = "c5.large"
    C5_XLARGE = "c5.xlarge"
    C5_2XLARGE = "c5.2xlarge"
    C5_4XLARGE = "c5.4xlarge"

    # GPU instances (T4)
    G4DN_XLARGE = "g4dn.xlarge"
    G4DN_2XLARGE = "g4dn.2xlarge"
    G4DN_4XLARGE = "g4dn.4xlarge"
    G4DN_12XLARGE = "g4dn.12xlarge"

    # GPU instances (A10G)
    G5_XLARGE = "g5.xlarge"
    G5_2XLARGE = "g5.2xlarge"
    G5_4XLARGE = "g5.4xlarge"
    G5_12XLARGE = "g5.12xlarge"

    # Let Batch choose
    OPTIMAL = "optimal"


# =============================================================================
# INSTANCE ASSIGNMENT
# =============================================================================

class InstanceAssignment(BaseModel):
    """
    Maps a service pattern to specific infrastructure.

    Uses pattern matching:
    - "od_*" matches all OD services (od_football_v2, od_cricket_v1, etc.)
    - "camera_view" matches exactly camera_view
    - "*" matches everything (default/fallback)
    """
    # Pattern matching
    service_pattern: str = Field(
        ...,
        description="Service ID pattern (supports * wildcard). Examples: 'od_*', 'camera_view', '*'"
    )

    # AWS Batch queue
    queue: str = Field(..., description="AWS Batch queue name")

    # Instance configuration
    instance_type: str = Field(
        default="optimal",
        description="EC2 instance type or 'optimal' for Batch to choose"
    )

    # Resource requirements
    vcpus: int = Field(default=0, ge=0, description="vCPU requirement (0 = use job definition default)")
    memory_mb: int = Field(default=0, ge=0, description="Memory in MB (0 = use job definition default)")
    gpu_count: int = Field(default=0, ge=0, description="GPU count (0 = no GPU required)")

    # Priority (higher = matched first when multiple patterns match)
    priority: int = Field(default=0, description="Match priority (higher = checked first)")

    class Config:
        use_enum_values = True

# =============================================================================
# DEPLOYMENT PROFILE
# =============================================================================

class DeploymentProfile(BaseModel):
    """
    Deployment profile configuration.

    Defines infrastructure settings:
    - Development: Local testing with file output
    - Production: AWS DynamoDB, S3, etc.
    """
    # Identity
    profile_id: str = Field(..., description="Unique profile identifier")
    profile_name: str = Field(default="", description="Human-readable name")
    environment: Environment = Field(default=Environment.PRODUCTION)

    # Development settings (local testing)
    local_output_dir: Optional[str] = Field(
        None,
        description="Local output directory for development (if set, skips DynamoDB)"
    )

    # Production settings (AWS)
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for DynamoDB and other services"
    )

    # DynamoDB write settings
    db_batch_size: int = Field(
        default=25,
        description="DynamoDB batch write size (max 25 per batch_writer)"
    )
    db_flush_interval_ms: int = Field(
        default=250,
        description="Flush interval for buffered writes"
    )

    # Orchestrator timing
    hls_metadata_head_start_seconds: int = Field(
        default=30,
        description="Seconds to wait after HLS metadata starts before other services"
    )

    class Config:
        use_enum_values = True

    def is_development(self) -> bool:
        """Check if this is a development profile"""
        return self.environment == Environment.DEVELOPMENT or self.local_output_dir is not None

    def is_production(self) -> bool:
        """Check if this is a production profile"""
        return self.environment == Environment.PRODUCTION and self.local_output_dir is None


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_development_profile(
    local_output_dir: str = "./test_output"
) -> DeploymentProfile:
    """
    Create a development profile for local testing.

    All output goes to local files instead of DynamoDB.
    """
    return DeploymentProfile(
        profile_id="development",
        profile_name="Development",
        environment=Environment.DEVELOPMENT,
        local_output_dir=local_output_dir,
        hls_metadata_head_start_seconds=10,  # Shorter delay for dev
    )


def create_production_profile(
    aws_region: str = "us-east-1"
) -> DeploymentProfile:
    """
    Create a production profile for AWS deployment.

    Output goes to DynamoDB tables.
    """
    return DeploymentProfile(
        profile_id="production",
        profile_name="Production",
        environment=Environment.PRODUCTION,
        aws_region=aws_region,
        db_batch_size=25,
        db_flush_interval_ms=250,
        hls_metadata_head_start_seconds=30,
    )
