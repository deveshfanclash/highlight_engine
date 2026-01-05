"""
Orchestrator Module

Two modes of operation:

1. Local Mode (MatchOrchestrator):
   - Spawns services as subprocesses on single machine
   - Good for development/testing

2. Distributed Mode (batch_submitter):
   - Submits jobs to AWS Batch
   - Each service runs on separate EC2 instance
   - Production-ready scaling
"""

from orchestrator.match_orchestrator import MatchOrchestrator
from orchestrator.batch_submitter import (
    submit_match_jobs,
    check_match_status,
    get_resume_position,
    BatchConfig,
)

__all__ = [
    # Local mode
    "MatchOrchestrator",
    # Distributed mode
    "submit_match_jobs",
    "check_match_status",
    "get_resume_position",
    "BatchConfig",
]
