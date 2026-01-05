"""
Orchestrator Module

Manages service lifecycle for a match:
- Loads configuration
- Spawns services based on config
- Monitors service health
- Handles graceful shutdown
"""

from orchestrator.match_orchestrator import MatchOrchestrator

__all__ = ["MatchOrchestrator"]
