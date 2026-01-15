# DynamoDB Writer & HLS Metadata Service Architecture

> **Reference Document**: Analysis and refactoring of the DynamoDB writer and HLS stream metadata service.
> **Date**: January 2025
> **Status**: High & Medium priority items completed

---

## Table of Contents

1. [Original Prompt](#original-prompt)
2. [Executive Summary](#executive-summary)
3. [First Principles Analysis](#first-principles-analysis)
4. [Problems Identified](#problems-identified)
5. [Architecture Before](#architecture-before)
6. [Solutions Implemented](#solutions-implemented)
7. [Architecture After](#architecture-after)
8. [File Changes Summary](#file-changes-summary)
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

### Key Components

| Component | Purpose | File |
|-----------|---------|------|
| **SyncDynamoDBWriter** | Synchronous buffered writes with retry | `db/dynamo.py` |
| **AsyncDynamoDBWriter** | Background thread queue for real-time writes | `db/dynamo.py` |
| **MetadataRepository** | Query abstraction for resume positions | `db/metadata_repository.py` |
| **HLSMetadataService** | Extract PTS times from HLS segments | `services/hls_metadata_service/service.py` |

---

## First Principles Analysis

### Core Responsibility of Each Component

**DynamoDB Writer**: Buffer and batch-write items to DynamoDB efficiently while handling AWS constraints.

**HLS Metadata Service**: Poll HLS playlist, extract PTS times via ffprobe, store frame metadata.

**Resume Logic**: Query stored metadata to determine where to resume processing after failure.

### Original Code Mixed Concerns

The original `DynamoDBWriter` class mixed:
1. **Data Preparation** - Float→Decimal conversion, timestamp injection
2. **Buffering/Batching** - Time & size-based flush logic
3. **Write Execution** - DynamoDB API calls, background threading
4. **Schema Knowledge** - Convenience methods embedded item structure

The original `HLSMetadataService` mixed:
1. **HLS Polling** - m3u8 fetching, segment tracking
2. **Metadata Extraction** - ffprobe subprocess, JSON parsing
3. **Persistence** - Batch writing to DynamoDB
4. **Service Lifecycle** - Signal handling, startup/shutdown

---

## Problems Identified

### High Priority (Bugs/Risks)

| Issue | Severity | Description |
|-------|----------|-------------|
| **No retry on write failure** | High | DynamoDB throttling causes data loss |
| **Schema inconsistency** | Medium | Used `pk`/`sk` but table uses `match_id`/`ptstime` |
| **Thread safety unclear** | Medium | Two modes in one class, implicit contracts |

### Medium Priority (Architecture)

| Issue | Description |
|-------|-------------|
| **No connection reuse** | `get_resume_position()` creates new connection each call |
| **Wasteful background thread** | HLS service started background thread but never used queue |
| **No observability** | No metrics for debugging production issues |

### Lower Priority (Deferred)

| Issue | Description |
|-------|-------------|
| **Code duplication** | `LocalFileWriter` duplicates `DynamoDBWriter` logic |
| **Convenience methods couple schema** | `write_inference_result()` embeds schema in writer |
| **BatchBuffer not extracted** | Batching logic not reusable |

---

## Architecture Before

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL ARCHITECTURE (COUPLED)                      │
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
│  │  write_inference_result(), write_camera_view_result()           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ HLSMetadataService                                              │   │
│  │                                                                 │   │
│  │  - Creates DynamoDBWriter internally (OWNS)                     │   │
│  │  - Starts background thread (NEVER USES QUEUE)                  │   │
│  │  - Writes items with pk/sk (WRONG KEYS)                         │   │
│  │  - Uses overwrite_keys=["pk", "sk"] (INCONSISTENT)              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ get_resume_position() - ORPHAN FUNCTION                         │   │
│  │                                                                 │   │
│  │  - Creates new boto3 connection each call                       │   │
│  │  - Queries on "match_id" (correct)                              │   │
│  │  - Lives in hls_metadata_service/service.py                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### DynamoDB Table Schemas (Discovery)

| Table | Partition Key | Sort Key | Used By |
|-------|--------------|----------|---------|
| `video_frames_metadata` | `match_id` | `ptstime` | HLS Metadata |
| `inference_results` | `{match_id}#{service_id}` | `frame_number` | OD, Camera |
| `inference_jobs` | `{match_id}#jobs` | `service_id` | Orchestrator |

---

## Solutions Implemented

### H1: Retry Logic with Exponential Backoff

**File**: `db/dynamo.py`

```python
RETRYABLE_ERROR_CODES = frozenset([
    'ProvisionedThroughputExceededException',
    'ThrottlingException',
    'InternalServerError',
    'ServiceUnavailable',
])

def _write_batch_with_retry(self, items, overwrite_keys=None):
    for attempt in range(self.config.max_retries + 1):
        try:
            return self._do_batch_write(items, overwrite_keys)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in RETRYABLE_ERROR_CODES and attempt < max_retries:
                delay = self._calculate_backoff(attempt)  # 100ms, 200ms, 400ms...
                time.sleep(delay)
                continue
            raise
```

**Configuration**:
- `max_retries`: 3 (default)
- `base_retry_delay_ms`: 100ms (default)
- Exponential backoff with jitter

### H2: Schema Consistency

**File**: `services/hls_metadata_service/service.py`

Before:
```python
item = {
    "pk": self.config.match_id,      # WRONG - redundant
    "sk": float(pts_time),           # WRONG - redundant
    "match_id": self.config.match_id,
    "ptstime": float(pts_time),
    ...
}
self._db_writer.write_items_batch(items, overwrite_keys=["pk", "sk"])  # WRONG
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

### H3: Thread Safety Documentation

**File**: `db/dynamo.py`

```python
class SyncDynamoDBWriter:
    """
    Synchronous DynamoDB writer with buffering.

    THREAD SAFETY: NOT thread-safe. Use from a single thread only.
    """

class AsyncDynamoDBWriter:
    """
    Asynchronous DynamoDB writer with background thread.

    THREAD SAFETY:
    - queue_item() is thread-safe (can be called from any thread)
    - Background thread handles all actual writes
    - Do NOT call write_item() directly - use queue_item()
    """
```

### M1: MetadataRepository

**New File**: `db/metadata_repository.py`

```python
class MetadataRepository:
    """
    Repository for HLS frame metadata queries.

    Features:
    - Connection reuse (lazy initialization)
    - Proper edge case handling
    - Abstraction over DynamoDB specifics
    """

    def __init__(self, table_name="video_frames_metadata"):
        self._table = None  # Lazy loaded

    def get_resume_position(self, match_id, fps=25.0, segment_length=7):
        """
        Get position to resume from.

        Strategy:
        1. Query recent frames
        2. Find second-highest complete segment (safety margin)
        3. Return position after last frame in that segment

        Edge cases handled:
        - No data → start position
        - Single segment → use that segment
        - Query failure → start position (fail safe)
        """
```

### M2: Sync/Async Writer Split

**File**: `db/dynamo.py`

| Class | Thread Safety | Use Case |
|-------|--------------|----------|
| `SyncDynamoDBWriter` | NOT thread-safe | HLS metadata (batch writes) |
| `AsyncDynamoDBWriter` | `queue_item()` is thread-safe | Real-time inference |
| `LocalFileWriter` | Supports both modes | Testing |

```python
# Factory function
def create_writer(table_name, use_background=True, local_output_dir=None, ...):
    if local_output_dir:
        writer = LocalFileWriter(...)
    elif use_background:
        writer = AsyncDynamoDBWriter(config)
        writer.start()
    else:
        writer = SyncDynamoDBWriter(config)
    return writer
```

### M3: Basic Metrics

**File**: `db/dynamo.py`

```python
@dataclass
class WriterMetrics:
    items_written: int = 0
    items_failed: int = 0
    batches_written: int = 0
    batches_failed: int = 0
    total_write_time_ms: float = 0.0
    retries: int = 0

    def summary(self) -> str:
        return (
            f"items_written={self.items_written}, "
            f"items_failed={self.items_failed}, "
            f"success_rate={self.success_rate:.1%}, "
            f"avg_latency={self.avg_latency_ms:.1f}ms, "
            f"retries={self.retries}"
        )
```

Logged on `stop()`:
```
INFO - SyncDynamoDBWriter stopped. Metrics: items_written=15000, items_failed=0, success_rate=100.0%, avg_latency=45.2ms, retries=2
```

---

## Architecture After

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NEW ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  WRITERS (db/dynamo.py)                                                │
│  ──────────────────────                                                │
│  ┌─────────────────────┐    ┌─────────────────────────────┐            │
│  │ SyncDynamoDBWriter  │    │ AsyncDynamoDBWriter         │            │
│  │                     │    │                             │            │
│  │ - NOT thread-safe   │    │ - queue_item() thread-safe  │            │
│  │ - Buffered writes   │    │ - Background thread         │            │
│  │ - Retry w/ backoff  │    │ - Wraps SyncDynamoDBWriter  │            │
│  │ - Metrics on stop   │    │ - Metrics on stop           │            │
│  └─────────────────────┘    └─────────────────────────────┘            │
│            ↑                              ↑                             │
│            │                              │                             │
│  ┌─────────┴───────────┐    ┌─────────────┴───────────────┐            │
│  │ HLSMetadataService  │    │ BaseService (OD, Camera)    │            │
│  │ (Sync batches)      │    │ (Async queue)               │            │
│  └─────────────────────┘    └─────────────────────────────┘            │
│                                                                         │
│  REPOSITORY (db/metadata_repository.py)                                │
│  ──────────────────────────────────────                                │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ MetadataRepository                                      │           │
│  │ - Lazy connection (reused)                              │           │
│  │ - get_resume_position() with edge case handling         │           │
│  └─────────────────────────────────────────────────────────┘           │
│            ↑                                                            │
│            │                                                            │
│  ┌─────────┴───────────────────────────────────────────────┐           │
│  │ Orchestrators (distributed, match, batch_submitter)     │           │
│  └─────────────────────────────────────────────────────────┘           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. HLSMetadataService starts
   └── Creates SyncDynamoDBWriter (no background thread)
   └── Polls m3u8 playlist
   └── For each .ts segment:
       └── ffprobe extracts PTS times
       └── Builds items: {match_id, ptstime, frame_number, segment_number, ...}
       └── Batch writes with retry to video_frames_metadata table

2. Orchestrator queries resume position
   └── MetadataRepository.get_resume_position(match_id)
   └── Queries video_frames_metadata table (connection reused)
   └── Returns {segment_number, frame_number}

3. Inference services start from resume position
   └── Creates AsyncDynamoDBWriter (background thread)
   └── Processes frames, calls queue_item() for each result
   └── Background thread batches and writes with retry
   └── On stop, logs metrics summary
```

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `db/dynamo.py` | **Rewritten** | Split into Sync/Async writers, added retry, metrics |
| `db/metadata_repository.py` | **New** | Repository for resume position queries |
| `db/__init__.py` | **Updated** | Export new classes |
| `services/hls_metadata_service/service.py` | **Updated** | Use SyncDynamoDBWriter, fix schema |
| `services/base_service.py` | **Updated** | Import new writer classes |
| `orchestrator/distributed_orchestrator.py` | **Updated** | Use MetadataRepository |
| `orchestrator/match_orchestrator.py` | **Updated** | Use MetadataRepository |
| `orchestrator/batch_submitter.py` | **Updated** | Use MetadataRepository |

---

## Usage Examples

### Using SyncDynamoDBWriter (HLS Metadata Pattern)

```python
from db.dynamo import SyncDynamoDBWriter, DynamoDBWriterConfig

config = DynamoDBWriterConfig.from_env("video_frames_metadata")
writer = SyncDynamoDBWriter(config)

# Write batch of items
items = [
    {"match_id": "match_123", "ptstime": 1.0, "frame_number": 1, ...},
    {"match_id": "match_123", "ptstime": 1.04, "frame_number": 2, ...},
]
writer.write_items_batch(items, overwrite_keys=["match_id", "ptstime"])

# Stop and log metrics
writer.stop()
# INFO - SyncDynamoDBWriter stopped. Metrics: items_written=2, ...
```

### Using AsyncDynamoDBWriter (Inference Pattern)

```python
from db.dynamo import AsyncDynamoDBWriter, DynamoDBWriterConfig

config = DynamoDBWriterConfig.from_env("inference_results")
writer = AsyncDynamoDBWriter(config)
writer.start()

# Queue items from any thread
for frame in frames:
    result = model.predict(frame)
    writer.queue_item({
        "pk": f"{match_id}#{service_id}",
        "sk": frame_number,
        "detections": result,
    })

# Stop and log metrics
writer.stop()
```

### Using Factory Function

```python
from db.dynamo import create_writer

# For testing (local files)
writer = create_writer(
    local_output_dir="/tmp/test",
    match_id="match_123",
    service_id="od_football"
)

# For production (DynamoDB with background thread)
writer = create_writer(
    table_name="inference_results",
    use_background=True
)

# For HLS metadata (DynamoDB, sync)
writer = create_writer(
    table_name="video_frames_metadata",
    use_background=False
)
```

### Using MetadataRepository

```python
from db.metadata_repository import MetadataRepository, get_resume_position

# Direct function call (convenience)
position = get_resume_position(
    match_id="match_123",
    stream_url="https://example.com/stream.m3u8"
)
# Returns: {"segment_number": 45, "frame_number": 1125}

# Repository pattern (connection reuse)
repo = MetadataRepository(table_name="video_frames_metadata")
position = repo.get_resume_position("match_123")
# Can make multiple queries with same connection
```

---

## Future Considerations

### Deferred (Lower Priority)

| Item | Description | Effort |
|------|-------------|--------|
| **Extract BatchBuffer** | Generic reusable batching component | Medium |
| **Schema Builders** | Move pk/sk construction to dedicated classes | Low |
| **Full DI** | Inject writers into services | Medium |
| **Extract FFProbeExtractor** | Separate extraction from service | Low |
| **Extract M3U8Poller** | Separate polling from service | Low |

### Potential Improvements

1. **Periodic metrics logging**: Currently only on stop; could add interval-based logging
2. **Circuit breaker**: After N consecutive failures, stop trying for a cooldown period
3. **Dead letter queue**: Store failed items for later retry/analysis
4. **Prometheus/CloudWatch integration**: Export metrics to monitoring systems

### Known Limitations

1. **Data loss on crash**: Items in buffer are lost if process crashes before flush
2. **No exactly-once semantics**: Retries may cause duplicates (idempotent keys help)
3. **No backpressure**: Queue can grow unbounded if writes are slower than production

---

## Configuration Reference

### DynamoDBWriterConfig

| Field | Default | Description |
|-------|---------|-------------|
| `table_name` | Required | DynamoDB table name |
| `region` | `us-east-1` | AWS region |
| `aws_access_key` | `None` | AWS access key (or use IAM role) |
| `aws_secret_key` | `None` | AWS secret key (or use IAM role) |
| `batch_size` | `12` | Items to buffer before flush |
| `flush_interval_ms` | `250` | Max time between flushes |
| `max_retries` | `3` | Retry attempts for transient errors |
| `base_retry_delay_ms` | `100` | Base delay for exponential backoff |
| `queue_timeout` | `5.0` | Queue.get() timeout (AsyncDynamoDBWriter) |

### Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `AWS_REGION` | All | AWS region |
| `AWS_ACCESS_KEY_DB` | All | AWS access key for DynamoDB |
| `AWS_SECRET_KEY_DB` | All | AWS secret key for DynamoDB |

---

## Appendix: Retry Behavior

### Retryable Errors

| Error Code | Description | Typical Cause |
|------------|-------------|---------------|
| `ProvisionedThroughputExceededException` | Exceeded table capacity | High write volume |
| `ThrottlingException` | Request throttled | Rate limiting |
| `InternalServerError` | DynamoDB internal error | AWS issue |
| `ServiceUnavailable` | Service temporarily unavailable | AWS issue |

### Backoff Calculation

```python
def _calculate_backoff(self, attempt: int) -> float:
    """Exponential backoff with jitter"""
    base_ms = self.config.base_retry_delay_ms * (2 ** attempt)
    jitter_ms = random.randint(0, base_ms // 2)
    return (base_ms + jitter_ms) / 1000.0

# Example with base_retry_delay_ms=100:
# Attempt 0: 100-150ms
# Attempt 1: 200-300ms
# Attempt 2: 400-600ms
```

---

*Document generated from architectural analysis session, January 2025*
