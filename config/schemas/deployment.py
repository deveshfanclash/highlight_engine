"""
Deployment Profile Configuration (Tier 3)

Defines HOW to run services on infrastructure.
Completely separate from game logic - same profile can run any sport.

This is the "infrastructure config" that DevOps manages.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class Environment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
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
# BATCH CONFIGURATION
# =============================================================================

class BatchQueueConfig(BaseModel):
    """Configuration for an AWS Batch queue"""
    queue_name: str
    compute_environment: str
    priority: int = Field(default=1)
    is_gpu: bool = Field(default=False)


class BatchConfig(BaseModel):
    """AWS Batch infrastructure configuration"""
    # Queue definitions
    queues: List[BatchQueueConfig] = Field(
        default_factory=list,
        description="Available Batch queues"
    )

    # Default queues (shortcuts)
    default_cpu_queue: str = Field(default="inference-cpu-queue")
    default_gpu_queue: str = Field(default="inference-gpu-queue")

    # Job definitions (service_type → job definition name)
    job_definitions: Dict[str, str] = Field(
        default_factory=lambda: {
            "hls_metadata": "hls-metadata-service",
            "camera_view": "camera-view-service",
            "object_detection": "od-service",
            "segmentation": "segmentation-service",
            "pose_estimation": "pose-service",
            "audio_analysis": "audio-service",
        }
    )

    # Job prefix for naming
    job_name_prefix: str = Field(default="inference")


# =============================================================================
# SCALING CONFIGURATION
# =============================================================================

class ScalingConfig(BaseModel):
    """Auto-scaling configuration for compute environments"""
    min_vcpus: int = Field(default=0, description="Minimum vCPUs to keep warm")
    max_vcpus: int = Field(default=256, description="Maximum vCPUs to scale to")
    desired_vcpus: int = Field(default=0, description="Desired vCPUs at rest")

    # Scaling triggers
    scale_up_threshold_jobs: int = Field(
        default=5,
        description="Scale up when this many jobs are pending"
    )
    scale_down_delay_seconds: int = Field(
        default=300,
        description="Wait this long before scaling down idle instances"
    )

    # Spot vs On-Demand
    use_spot_instances: bool = Field(default=False)
    spot_bid_percentage: int = Field(
        default=100,
        ge=1, le=100,
        description="Max bid as percentage of on-demand price"
    )


# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration"""
    # Health checks
    heartbeat_timeout_seconds: int = Field(
        default=300,
        description="Mark job as stuck if no heartbeat for this long"
    )
    job_timeout_minutes: int = Field(
        default=180,
        description="Maximum job duration before termination"
    )

    # Retry policy
    max_retries: int = Field(default=3, description="Maximum restart attempts")
    retry_delay_seconds: int = Field(default=30, description="Delay between retries")

    # Alerting
    enable_alerts: bool = Field(default=True)
    alert_sns_topic_arn: Optional[str] = Field(None, description="SNS topic for alerts")
    alert_on_failure: bool = Field(default=True)
    alert_on_stuck: bool = Field(default=True)
    alert_on_retry_exhausted: bool = Field(default=True)

    # Metrics
    enable_cloudwatch_metrics: bool = Field(default=True)
    metrics_namespace: str = Field(default="Inference/Jobs")


# =============================================================================
# COST CONTROLS
# =============================================================================

class CostControlConfig(BaseModel):
    """Cost control and limits"""
    # Concurrency limits
    max_concurrent_matches: int = Field(
        default=50,
        description="Maximum matches processing simultaneously"
    )
    max_concurrent_jobs_per_match: int = Field(
        default=10,
        description="Maximum jobs per match"
    )

    # Instance preferences
    prefer_spot_instances: bool = Field(default=True)
    allowed_instance_families: List[str] = Field(
        default_factory=lambda: ["c5", "g4dn", "g5"],
        description="Allowed EC2 instance families"
    )

    # Budget controls
    enable_budget_alerts: bool = Field(default=False)
    daily_budget_usd: Optional[float] = Field(None, description="Daily budget limit")
    monthly_budget_usd: Optional[float] = Field(None, description="Monthly budget limit")


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

class DatabaseConfig(BaseModel):
    """Database configuration for this deployment"""
    # DynamoDB tables
    inference_results_table: str = Field(default="inference_results")
    jobs_tracking_table: str = Field(default="inference_jobs")
    metadata_table: str = Field(default="video_frames_metadata")

    # Write settings
    batch_size: int = Field(default=25, description="DynamoDB batch write size")
    flush_interval_ms: int = Field(default=250, description="Flush interval")

    # S3 for large outputs (masks, etc.)
    output_bucket: Optional[str] = Field(None, description="S3 bucket for large outputs")
    output_prefix: str = Field(default="inference-outputs/")


# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================

class NetworkConfig(BaseModel):
    """Network configuration for services"""
    vpc_id: Optional[str] = None
    subnet_ids: List[str] = Field(default_factory=list)
    security_group_ids: List[str] = Field(default_factory=list)

    # Endpoints
    use_vpc_endpoints: bool = Field(
        default=True,
        description="Use VPC endpoints for S3/DynamoDB (reduces cost)"
    )


# =============================================================================
# DEPLOYMENT PROFILE
# =============================================================================

class DeploymentProfile(BaseModel):
    """
    Complete deployment profile configuration.

    This defines HOW to run inference services on AWS infrastructure.
    Completely independent of game logic - same profile can run any sport.

    Typical profiles:
    - development: Local/single instance, no scaling
    - staging: Small scale, spot instances
    - production: Full scale, mixed spot/on-demand
    - production_multi_gpu: Multiple GPU queues for parallel models
    """
    # Identity
    profile_id: str = Field(..., description="Unique profile identifier")
    profile_name: str = Field(default="", description="Human-readable name")
    description: str = Field(default="", description="Profile description")
    environment: Environment = Field(default=Environment.PRODUCTION)

    # Instance assignments (the core mapping)
    instance_assignments: List[InstanceAssignment] = Field(
        default_factory=list,
        description="Service → Instance mappings"
    )

    # AWS Batch configuration
    batch: BatchConfig = Field(default_factory=BatchConfig)

    # Operational settings
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    cost_controls: CostControlConfig = Field(default_factory=CostControlConfig)

    # Infrastructure
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)

    # Timing
    hls_metadata_head_start_seconds: int = Field(
        default=30,
        description="Delay after HLS metadata starts before other services"
    )

    class Config:
        use_enum_values = True

    def get_instance_assignment(self, service_id: str) -> Optional[InstanceAssignment]:
        """
        Get instance assignment for a service ID.

        Matches against patterns in priority order.
        Returns None if no pattern matches (use defaults).
        """
        import fnmatch

        # Sort by priority (highest first)
        sorted_assignments = sorted(
            self.instance_assignments,
            key=lambda x: x.priority,
            reverse=True
        )

        for assignment in sorted_assignments:
            if fnmatch.fnmatch(service_id, assignment.service_pattern):
                return assignment

        return None

    def get_queue_for_service(self, service_id: str, requires_gpu: bool = False) -> str:
        """Get the appropriate queue for a service"""
        assignment = self.get_instance_assignment(service_id)
        if assignment:
            return assignment.queue

        # Fallback to defaults
        return self.batch.default_gpu_queue if requires_gpu else self.batch.default_cpu_queue


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_development_profile() -> DeploymentProfile:
    """Create a development profile for local testing"""
    return DeploymentProfile(
        profile_id="development",
        profile_name="Development",
        description="Local development and testing",
        environment=Environment.DEVELOPMENT,
        instance_assignments=[
            InstanceAssignment(
                service_pattern="*",
                queue="local-queue",
                instance_type="optimal",
                vcpus=2,
                memory_mb=4096,
            ),
        ],
        scaling=ScalingConfig(
            max_vcpus=8,
            use_spot_instances=False,
        ),
        monitoring=MonitoringConfig(
            enable_alerts=False,
            enable_cloudwatch_metrics=False,
        ),
        cost_controls=CostControlConfig(
            max_concurrent_matches=2,
        ),
    )


def create_production_profile() -> DeploymentProfile:
    """Create a standard production profile"""
    return DeploymentProfile(
        profile_id="production",
        profile_name="Production",
        description="Standard production deployment",
        environment=Environment.PRODUCTION,
        instance_assignments=[
            # CPU services
            InstanceAssignment(
                service_pattern="hls_metadata",
                queue="inference-cpu-queue",
                instance_type="c5.large",
                vcpus=2,
                memory_mb=4096,
                priority=10,
            ),
            InstanceAssignment(
                service_pattern="camera_view",
                queue="inference-cpu-queue",
                instance_type="c5.xlarge",
                vcpus=4,
                memory_mb=8192,
                priority=10,
            ),
            # GPU services
            InstanceAssignment(
                service_pattern="od_*",
                queue="inference-gpu-queue",
                instance_type="g4dn.xlarge",
                vcpus=4,
                memory_mb=16384,
                gpu_count=1,
                priority=5,
            ),
            InstanceAssignment(
                service_pattern="seg_*",
                queue="inference-gpu-queue",
                instance_type="g4dn.2xlarge",
                vcpus=8,
                memory_mb=32768,
                gpu_count=1,
                priority=5,
            ),
            # Default fallback
            InstanceAssignment(
                service_pattern="*",
                queue="inference-cpu-queue",
                instance_type="c5.large",
                vcpus=2,
                memory_mb=4096,
                priority=0,
            ),
        ],
        scaling=ScalingConfig(
            max_vcpus=256,
            use_spot_instances=True,
            spot_bid_percentage=80,
        ),
        monitoring=MonitoringConfig(
            enable_alerts=True,
            heartbeat_timeout_seconds=300,
            max_retries=3,
        ),
    )


def create_multi_gpu_profile() -> DeploymentProfile:
    """Create a multi-GPU production profile for parallel model inference"""
    return DeploymentProfile(
        profile_id="production_multi_gpu",
        profile_name="Production Multi-GPU",
        description="Multiple GPU queues for parallel model inference",
        environment=Environment.PRODUCTION,
        batch=BatchConfig(
            queues=[
                BatchQueueConfig(queue_name="inference-cpu-queue", compute_environment="cpu-env", is_gpu=False),
                BatchQueueConfig(queue_name="inference-gpu-queue-1", compute_environment="gpu-env-1", is_gpu=True),
                BatchQueueConfig(queue_name="inference-gpu-queue-2", compute_environment="gpu-env-2", is_gpu=True),
                BatchQueueConfig(queue_name="inference-gpu-queue-3", compute_environment="gpu-env-3", is_gpu=True),
            ],
        ),
        instance_assignments=[
            # CPU services
            InstanceAssignment(
                service_pattern="hls_metadata",
                queue="inference-cpu-queue",
                instance_type="c5.large",
                priority=10,
            ),
            InstanceAssignment(
                service_pattern="camera_view",
                queue="inference-cpu-queue",
                instance_type="c5.xlarge",
                priority=10,
            ),
            # First OD model → GPU queue 1
            InstanceAssignment(
                service_pattern="od_*_v1",
                queue="inference-gpu-queue-1",
                instance_type="g4dn.xlarge",
                gpu_count=1,
                priority=10,
            ),
            # Second OD model → GPU queue 2
            InstanceAssignment(
                service_pattern="od_*_v2",
                queue="inference-gpu-queue-2",
                instance_type="g4dn.xlarge",
                gpu_count=1,
                priority=10,
            ),
            # Segmentation → GPU queue 3
            InstanceAssignment(
                service_pattern="seg_*",
                queue="inference-gpu-queue-3",
                instance_type="g4dn.2xlarge",
                gpu_count=1,
                priority=10,
            ),
            # Default GPU
            InstanceAssignment(
                service_pattern="*",
                queue="inference-gpu-queue-1",
                instance_type="g4dn.xlarge",
                gpu_count=1,
                priority=0,
            ),
        ],
    )
