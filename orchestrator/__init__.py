"""
Orchestrator Module

Three modes of operation:

1. Local Mode (MatchOrchestrator):
   - Spawns services as subprocesses on single machine
   - Good for development/testing
   - Usage: python -m orchestrator --config config.yaml --match-id X --stream-url Y

2. Distributed Mode (batch_submitter):
   - Submits jobs to AWS Batch
   - Each service runs on separate EC2 instance
   - Production-ready scaling
   - Usage: submit_match_jobs(match_id, stream_url, game_config)

3. Distributed Mode with Instance Mapping (DistributedOrchestrator):
   - Advanced orchestration with explicit instance mapping
   - Service → EC2 instance type mapping
   - Job monitoring and auto-restart
   - Usage: DistributedOrchestrator(match_id, stream_url, config, instance_mapping)

4. Job Monitoring (JobMonitor):
   - Health checks for running jobs
   - Auto-restart failed/stuck jobs
   - Alerting via SNS
   - Usage: Lambda triggered by EventBridge every minute
"""

from orchestrator.match_orchestrator import MatchOrchestrator
from orchestrator.batch_submitter import (
    submit_match_jobs,
    check_match_status,
    get_resume_position,
    BatchConfig,
)
from orchestrator.distributed_orchestrator import (
    DistributedOrchestrator,
    submit_distributed_match,
    InstanceConfig,
    ServiceJobConfig,
    MatchJob,
)
from orchestrator.job_monitor import (
    JobMonitor,
    JobHealth,
    JobHealthReport,
    MatchHealthReport,
    MonitorConfig,
    monitor_handler,
)

__all__ = [
    # Local mode
    "MatchOrchestrator",

    # Basic distributed mode
    "submit_match_jobs",
    "check_match_status",
    "get_resume_position",
    "BatchConfig",

    # Advanced distributed mode with instance mapping
    "DistributedOrchestrator",
    "submit_distributed_match",
    "InstanceConfig",
    "ServiceJobConfig",
    "MatchJob",

    # Job monitoring
    "JobMonitor",
    "JobHealth",
    "JobHealthReport",
    "MatchHealthReport",
    "MonitorConfig",
    "monitor_handler",
]
