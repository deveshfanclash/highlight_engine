"""
HLS Stream Metadata Service

Extracts frame-level metadata (PTS times, segment mapping) from HLS streams.
This service runs BEFORE other services and provides data for resume functionality.
"""

from services.hls_metadata_service.service import HLSMetadataService, HLSMetadataServiceConfig

__all__ = ["HLSMetadataService", "HLSMetadataServiceConfig"]
