"""
Entry point for running Match Orchestrator as a module.

Usage:
    python -m orchestrator --match-id ABC --stream-url URL --config config.yaml
"""

from orchestrator.match_orchestrator import main

if __name__ == "__main__":
    main()
