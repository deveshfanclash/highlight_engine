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
    )
