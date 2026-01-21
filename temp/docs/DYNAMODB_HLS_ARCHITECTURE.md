# DynamoDB Writer & HLS Metadata Service Architecture

> **Reference Document**: Analysis and refactoring plan for the DynamoDB writer and HLS stream metadata service.
> **Date**: January 2025
> **Status**: Analysis complete, implementation plan ready

---

## Table of Contents

1. [Original Prompt](#original-prompt)
2. [Executive Summary](#executive-summary)
3. [First Principles Analysis](#first-principles-analysis)
4. [Current Architecture](#current-architecture)
5. [Problems Identified](#problems-identified)
6. [Proposed Solutions](#proposed-solutions)
7. [Implementation Plan](#implementation-plan)
8. [Code Changes Reference](#code-changes-reference)
9. [Usage Examples](#usage-examples)
10. [Future Considerations](#future-considerations)

---

## Original Prompt

```
Understand the DynamoDB writer. Assume you are an experienced Python design engineer.
You understand the fundamental principle of simplicity. You understand the importance
of extendable code and the crucial low level and high level design concepts.

Now understand the HLS stream metadata service where DynamoDB is an intricate part.
Understand how the DynamoDB code is written, its complexities etc...

Think from first principles, what the code is really trying to achieve. We need to
break down into simple blocks to clearly define responsibilities. We need to understand
all the complexity in order to create a simplified architecture of the current codebase.
```

---

## Executive Summary

### What the System Does

The system solves one core problem: **Mapping HLS video streams to queryable frame-level metadata for distributed video processing with fault-tolerant resume capability.**

```
HLS Stream (.ts segments) → Frame Metadata (DynamoDB) → Resume Position → Inference Services
```

### Key Files

| File | Purpose |
|------|---------|
| `db/dynamo.py` | DynamoDB writer with batching and background thread support |
| `db/metadata_repository.py` | Repository for resume position queries (NEW) |
| `services/hls_metadata_service/service.py` | Extract PTS times from HLS segments |
| `services/base_service.py` | Base class for inference services |
| `orchestrator/*.py` | Job orchestration and resume logic |

---

## First Principles Analysis

### What Each Component Is Fundamentally Trying to Achieve

**DynamoDB Writer**
- Buffer items to reduce API calls
- Batch writes for efficiency (DynamoDB limit: 25 items per batch)
- Handle AWS constraints (float→Decimal conversion)
- Support both sync and async patterns

**HLS Metadata Service**
- Poll m3u8 playlist for new segments
- Extract frame-level PTS times via ffprobe
- Store metadata enabling other services to resume

**Resume Logic**
- Query stored metadata
- Determine safe position to restart (avoid incomplete segments)
- Enable fault-tolerant distributed processing

### The Three Real Concerns in DynamoDB Writer

| Concern | Lines | What It Does |
|---------|-------|--------------|
| **Data Preparation** | 104-127 | Float→Decimal conversion, timestamp injection |
| **Buffering/Batching** | 129-156 | Time & size-based flush logic |
| **Write Execution** | 158-324 | DynamoDB API calls, background threading |

### The Four Real Concerns in HLS Metadata Service

| Concern | Lines | What It Does |
|---------|-------|--------------|
| **HLS Polling** | 87-137 | m3u8 fetching, segment tracking, timeout handling |
| **Metadata Extraction** | 138-236 | ffprobe subprocess, JSON parsing, frame data extraction |
| **Persistence** | 238-260 | Batch writing to DynamoDB or local files |
| **Service Lifecycle** | 262-311 | Signal handling, startup/shutdown coordination |

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CURRENT ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ DynamoDBWriter (MIXED CONCERNS)                                 │   │
│  │                                                                 │   │
│  │  MODE 1: Direct           MODE 2: Queue                         │   │
│  │  write_item() ────────►   queue_item() ───► Background Thread   │   │
│  │       │                        │                                │   │
│  │       ▼                        ▼                                │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │ Shared buffer, no retry, implicit thread contract       │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  │  SCHEMA KNOWLEDGE EMBEDDED:                                     │   │
│  │  write_inference_result() - knows pk/sk format                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ HLSMetadataService                                              │   │
│  │                                                                 │   │
│  │  - Creates DynamoDBWriter internally                            │   │
│  │  - Starts background thread (but uses write_items_batch)        │   │
│  │  - Schema: pk/sk AND match_id/ptstime (redundant)               │   │
│  │  - overwrite_keys=["pk", "sk"] (should be match_id/ptstime)     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ get_resume_position() - STANDALONE FUNCTION                     │   │
│  │                                                                 │   │
│  │  - Creates new boto3 connection each call                       │   │
│  │  - Queries on "match_id" (correct key)                          │   │
│  │  - Lives in hls_metadata_service/service.py                     │   │
│  │  - Duplicated imports in multiple orchestrators                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### DynamoDB Table Schemas

| Table | Partition Key | Sort Key | Used By |
|-------|--------------|----------|---------|
| `video_frames_metadata` | `match_id` | `ptstime` | HLS Metadata |
| `inference_results` | `{match_id}#{service_id}` | `frame_number` | OD, Camera |
| `inference_jobs` | `{match_id}#jobs` | `service_id` | Orchestrator |

---

## Problems Identified

### High Priority (Bugs/Risks)

| Issue | Severity | Location | Description |
|-------|----------|----------|-------------|
| **No retry on write failure** | High | `db/dynamo.py:203-213` | DynamoDB throttling causes silent data loss |
| **Schema inconsistency** | Medium | `hls_metadata_service/service.py:197-206` | Uses `pk`/`sk` but table uses `match_id`/`ptstime` |
| **Thread safety unclear** | Medium | `db/dynamo.py:54-72` | Two modes in one class, implicit contracts |

### Medium Priority (Architecture)

| Issue | Location | Description |
|-------|----------|-------------|
| **No connection reuse** | `hls_metadata_service/service.py:354-360` | `get_resume_position()` creates new connection each call |
| **Wasteful background thread** | `hls_metadata_service/service.py:273-274` | Starts background thread but never uses queue |
| **No observability** | `db/dynamo.py` | No metrics for debugging production issues |

### Lower Priority (Deferred)

| Issue | Description |
|-------|-------------|
| Code duplication | `LocalFileWriter` duplicates `DynamoDBWriter` logic |
| Schema coupling | `write_inference_result()` embeds schema in writer |
| No BatchBuffer extraction | Batching logic not reusable |

---

## Proposed Solutions

### H1: Add Retry Logic with Exponential Backoff

**Problem**: Transient DynamoDB errors cause data loss.

**Solution**: Add retry with exponential backoff for retryable errors.

```python
RETRYABLE_ERROR_CODES = frozenset([
    'ProvisionedThroughputExceededException',
    'ThrottlingException',
    'InternalServerError',
    'ServiceUnavailable',
])

def _write_batch_with_retry(self, items, max_retries=3):
    from botocore.exceptions import ClientError

    for attempt in range(max_retries + 1):
        try:
            return self._do_batch_write(items)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in RETRYABLE_ERROR_CODES and attempt < max_retries:
                delay = self._calculate_backoff(attempt)
                time.sleep(delay)
                continue
            raise

def _calculate_backoff(self, attempt: int) -> float:
    """Exponential backoff with jitter: 100ms, 200ms, 400ms..."""
    base_ms = 100 * (2 ** attempt)
    jitter_ms = random.randint(0, base_ms // 2)
    return (base_ms + jitter_ms) / 1000.0
```

### H2: Fix Schema Consistency

**Problem**: HLS metadata uses `pk`/`sk` but table schema is `match_id`/`ptstime`.

**Solution**: Remove redundant fields, use correct overwrite keys.

Before:
```python
item = {
    "pk": self.config.match_id,      # Redundant
    "sk": float(pts_time),           # Redundant
    "match_id": self.config.match_id,
    "ptstime": float(pts_time),
    ...
}
self._db_writer.write_items_batch(items, overwrite_keys=["pk", "sk"])
```

After:
```python
item = {
    "match_id": self.config.match_id,  # Partition key
    "ptstime": float(pts_time),        # Sort key
    "segment": segment_name,
    "ts_frame": idx,
    "frame_number": self._frame_number,
    "segment_number": segment_number,
}
self._db_writer.write_items_batch(items, overwrite_keys=["match_id", "ptstime"])
```

### H3: Document Thread Safety Contract

**Problem**: Two operational modes in one class with implicit contracts.

**Solution**: Clear documentation and optional runtime checks.

```python
class DynamoDBWriter:
    """
    THREAD SAFETY:
    ─────────────
    1. DIRECT MODE (Single-threaded):
       - Call write_item() directly from one thread
       - Buffer is NOT thread-safe

    2. QUEUE MODE (Multi-threaded):
       - Call start_background_writer() first
       - Then use queue_item() from any thread (thread-safe)
       - Background thread handles all writes

    WARNING: Mixing modes leads to undefined behavior.
    """
```

### M1: Create MetadataRepository

**Problem**: `get_resume_position()` creates new connection each call, edge cases not handled.

**Solution**: Repository pattern with lazy connection and proper error handling.

```python
# db/metadata_repository.py

class MetadataRepository:
    """Repository for HLS frame metadata queries."""

    def __init__(self, table_name="video_frames_metadata"):
        self._table = None  # Lazy loaded

    def get_resume_position(self, match_id, fps=25.0, segment_length=7):
        """
        Get position to resume from.

        Edge cases handled:
        - No data → start position
        - Single segment → use that segment
        - Query failure → start position (fail safe)
        """
        try:
            return self._do_get_resume_position(match_id, fps, segment_length)
        except Exception as e:
            logger.error(f"Failed to get resume position: {e}")
            return ResumePosition.from_start()
```

### M2: Separate Sync vs Async Writer

**Problem**: HLS service starts background thread but doesn't use queue.

**Solution**: Two explicit writer classes.

```python
class SyncDynamoDBWriter:
    """
    Synchronous writer. NOT thread-safe.
    Use for: HLS metadata (batch writes), batch processing.
    """

class AsyncDynamoDBWriter:
    """
    Asynchronous writer with background thread.
    queue_item() is thread-safe.
    Use for: Real-time inference services.
    """
```

### M3: Add Basic Metrics

**Problem**: No visibility into writer performance.

**Solution**: Metrics dataclass logged on stop.

```python
@dataclass
class WriterMetrics:
    items_written: int = 0
    items_failed: int = 0
    retries: int = 0
    avg_latency_ms: float = 0.0

    def summary(self) -> str:
        return f"written={self.items_written}, failed={self.items_failed}, retries={self.retries}"
```

---

## Implementation Plan

### Phase 1: High Priority (Fix Bugs/Risks)

| Task | File | Effort |
|------|------|--------|
| H1: Add retry logic | `db/dynamo.py` | Medium |
| H2: Fix schema consistency | `services/hls_metadata_service/service.py` | Low |
| H3: Document thread safety | `db/dynamo.py` | Low |

### Phase 2: Medium Priority (Architecture)

| Task | File | Effort |
|------|------|--------|
| M1: Create MetadataRepository | `db/metadata_repository.py` (new) | Medium |
| M2: Sync/Async writer split | `db/dynamo.py` | Medium |
| M3: Add basic metrics | `db/dynamo.py` | Low |
| Update consumers | Multiple orchestrators | Low |

### Phase 3: Lower Priority (Deferred)

| Task | Description |
|------|-------------|
| Extract BatchBuffer | Generic reusable batching component |
| Schema Builders | Move pk/sk construction to dedicated classes |
| Full dependency injection | Services receive writers instead of creating |

---

## Code Changes Reference

### New File: `db/metadata_repository.py`

```python
"""
Metadata Repository

Encapsulates all read operations for HLS frame metadata.
"""

import os
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResumePosition:
    """Position to resume processing from"""
    segment_number: int
    frame_number: int

    @classmethod
    def from_start(cls) -> "ResumePosition":
        return cls(segment_number=1, frame_number=1)

    def to_dict(self) -> Dict[str, int]:
        return {
            "segment_number": self.segment_number,
            "frame_number": self.frame_number,
        }


class MetadataRepository:
    """Repository for HLS frame metadata queries."""

    def __init__(self, table_name: str = "video_frames_metadata"):
        self.table_name = table_name
        self._table = None

    def _get_table(self):
        """Lazy-load DynamoDB table"""
        if self._table is None:
            import boto3
            dynamodb = boto3.resource(
                'dynamodb',
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_DB"),
                aws_secret_access_key=os.getenv("AWS_SECRET_KEY_DB"),
            )
            self._table = dynamodb.Table(self.table_name)
        return self._table

    def get_resume_position(
        self,
        match_id: str,
        fps: float = 25.0,
        segment_length: int = 7
    ) -> ResumePosition:
        """Get position to resume processing from."""
        try:
            from boto3.dynamodb.conditions import Key
            table = self._get_table()

            limit = int(fps * segment_length * 2)
            response = table.query(
                KeyConditionExpression=Key("match_id").eq(match_id),
                ScanIndexForward=False,
                Limit=limit
            )

            items = response.get("Items", [])
            if not items:
                return ResumePosition.from_start()

            # Find second-highest segment (safety margin)
            segments = sorted({int(item["segment_number"]) for item in items})
            target = segments[-2] if len(segments) >= 2 else segments[-1]

            segment_items = [i for i in items if int(i["segment_number"]) == target]
            max_frame = max(segment_items, key=lambda x: int(x["frame_number"]))

            return ResumePosition(
                segment_number=target + 1,
                frame_number=int(max_frame["frame_number"]) + 1,
            )

        except Exception as e:
            logger.error(f"Failed to get resume position: {e}")
            return ResumePosition.from_start()


# Convenience function
def get_resume_position(
    match_id: str,
    stream_url: str,
    fps: float = 25.0,
    segment_length: int = 7,
    db_table_name: str = "video_frames_metadata"
) -> Dict[str, int]:
    """Get resume position (convenience function for orchestrators)."""
    repo = MetadataRepository(table_name=db_table_name)
    return repo.get_resume_position(match_id, fps, segment_length).to_dict()
```

### Updated: `db/dynamo.py` (Key Changes)

```python
# Add to DynamoDBWriterConfig
@dataclass
class DynamoDBWriterConfig:
    # ... existing fields ...
    max_retries: int = 3
    base_retry_delay_ms: int = 100


# Add metrics class
@dataclass
class WriterMetrics:
    items_written: int = 0
    items_failed: int = 0
    retries: int = 0

    def summary(self) -> str:
        return f"written={self.items_written}, failed={self.items_failed}, retries={self.retries}"


# Add to DynamoDBWriter class
RETRYABLE_ERRORS = frozenset([
    'ProvisionedThroughputExceededException',
    'ThrottlingException',
    'InternalServerError',
    'ServiceUnavailable',
])

def _write_batch_with_retry(self, items, overwrite_keys=None):
    from botocore.exceptions import ClientError

    for attempt in range(self.config.max_retries + 1):
        try:
            return self._do_batch_write(items, overwrite_keys)
        except ClientError as e:
            code = e.response.get('Error', {}).get('Code', '')
            if code in RETRYABLE_ERRORS and attempt < self.config.max_retries:
                self._metrics.retries += 1
                delay = (100 * (2 ** attempt) + random.randint(0, 50)) / 1000
                time.sleep(delay)
                continue
            raise

def stop(self, timeout: float = 10.0):
    # ... existing code ...
    logger.info(f"Writer stopped. {self._metrics.summary()}")
```

### Updated: `services/hls_metadata_service/service.py` (Key Changes)

```python
# Change item construction (remove pk/sk)
item = {
    "match_id": self.config.match_id,
    "ptstime": float(pts_time),
    "segment": segment_name,
    "ts_frame": idx,
    "frame_number": self._frame_number,
    "segment_number": segment_number,
}

# Change overwrite keys
self._db_writer.write_items_batch(
    items,
    overwrite_keys=["match_id", "ptstime"]  # Match table schema
)
```

### Updated: Orchestrator Imports

```python
# Before (in multiple files)
from services.hls_metadata_service.service import get_resume_position

# After
from db.metadata_repository import get_resume_position
```

---

## Usage Examples

### Using MetadataRepository

```python
from db.metadata_repository import MetadataRepository, get_resume_position

# Option 1: Convenience function
position = get_resume_position(
    match_id="match_123",
    stream_url="https://example.com/stream.m3u8"
)
print(position)  # {"segment_number": 45, "frame_number": 1125}

# Option 2: Repository (connection reuse)
repo = MetadataRepository()
pos1 = repo.get_resume_position("match_123")
pos2 = repo.get_resume_position("match_456")  # Same connection
```

### HLS Metadata Item Schema

```python
# Correct schema (matches DynamoDB table)
item = {
    "match_id": "match_123",        # Partition key
    "ptstime": 1.234,               # Sort key
    "segment": "segment_001.ts",
    "ts_frame": 0,
    "frame_number": 1,
    "segment_number": 1,
    "written_at": "2025-01-21T12:00:00Z"
}
```

---

## Future Considerations

### Potential Improvements

| Item | Description | Priority |
|------|-------------|----------|
| Circuit breaker | Stop retrying after N consecutive failures | Low |
| Dead letter queue | Store failed items for later analysis | Low |
| Prometheus metrics | Export to monitoring systems | Medium |
| Async/await refactor | Replace threads with asyncio | Low |

### Known Limitations

1. **Data loss on crash**: Items in buffer are lost if process crashes
2. **No exactly-once**: Retries may cause duplicates (idempotent keys help)
3. **No backpressure**: Queue can grow unbounded

### Architecture Target State

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TARGET ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  WRITERS (db/dynamo.py)                                                │
│  ┌─────────────────────┐    ┌─────────────────────────────┐            │
│  │ SyncDynamoDBWriter  │    │ AsyncDynamoDBWriter         │            │
│  │ - NOT thread-safe   │    │ - queue_item() thread-safe  │            │
│  │ - Retry w/ backoff  │    │ - Wraps SyncDynamoDBWriter  │            │
│  │ - Metrics on stop   │    │ - Metrics on stop           │            │
│  └─────────────────────┘    └─────────────────────────────┘            │
│            ↑                              ↑                             │
│  ┌─────────┴───────────┐    ┌─────────────┴───────────────┐            │
│  │ HLSMetadataService  │    │ BaseService (OD, Camera)    │            │
│  └─────────────────────┘    └─────────────────────────────┘            │
│                                                                         │
│  REPOSITORY (db/metadata_repository.py)                                │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ MetadataRepository                                      │           │
│  │ - Lazy connection, get_resume_position()                │           │
│  └─────────────────────────────────────────────────────────┘           │
│            ↑                                                            │
│  ┌─────────┴───────────────────────────────────────────────┐           │
│  │ Orchestrators (distributed, match, batch_submitter)     │           │
│  └─────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix: Configuration Reference

### DynamoDBWriterConfig Fields

| Field | Default | Description |
|-------|---------|-------------|
| `table_name` | Required | DynamoDB table name |
| `region` | `us-east-1` | AWS region |
| `batch_size` | `12` | Items to buffer before flush |
| `flush_interval_ms` | `250` | Max time between flushes |
| `max_retries` | `3` | Retry attempts (proposed) |
| `base_retry_delay_ms` | `100` | Base delay for backoff (proposed) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AWS_REGION` | AWS region |
| `AWS_ACCESS_KEY_DB` | AWS access key |
| `AWS_SECRET_KEY_DB` | AWS secret key |

---

*Document generated from architectural analysis session, January 2025*
