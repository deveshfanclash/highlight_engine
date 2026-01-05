"""
Entry point for running OD service as a module.

Usage:
    python -m services.od_service --match-id ABC --stream-url URL --model-id football_v2
"""

from services.od_service.service import main

if __name__ == "__main__":
    main()
