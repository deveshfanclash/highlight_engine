"""
Entry point for running Camera View service as a module.

Usage:
    python -m services.camera_view_service --match-id ABC --stream-url URL
"""

from services.camera_view_service.service import main

if __name__ == "__main__":
    main()
