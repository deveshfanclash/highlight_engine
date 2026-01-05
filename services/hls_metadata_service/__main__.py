"""
Entry point for running HLS Metadata service as a module.

Usage:
    python -m services.hls_metadata_service --match-id ABC --stream-url URL
"""

from services.hls_metadata_service.service import main

if __name__ == "__main__":
    main()
