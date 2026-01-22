# Multi-Worker Support Refactoring Plan

## Overview

Add multi-worker parallel processing to the inference system for true real-time performance. This refactoring introduces process-based workers for GPU parallelism, automatic stream buffering, and batch-level frame distribution.

**Key Changes:**
1. Add `num_workers` and `queue_size` to config hierarchy
2. Make `StreamBuffer` default for HLS/RTSP streams
3. Implement batch-level distribution (`batch_id % num_workers`)
4. Add `WorkerPool` for process-based parallel inference

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           MAIN PROCESS                               │
│                                                                      │
│  ┌──────────────┐     ┌────────────────┐     ┌──────────────────┐  │
│  │FrameProvider │────▶│ FrameDistributor│────▶│ Worker Queues    │  │
│  │ + StreamBuffer│     │ (batch-level   │     │ [Q0, Q1, Q2, Q3] │  │
│  │ (auto for HLS)│     │  round-robin)  │     │                  │  │
│  └──────────────┘     └────────────────┘     └────────┬─────────┘  │
│                                                        │            │
│                       ┌────────────────────────────────┼──────────┐ │
│                       ▼              ▼              ▼              ▼ │
│  WORKER PROCESSES: ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│                    │Worker 0 │ │Worker 1 │ │Worker 2 │ │Worker 3 │ │
│                    │(Process)│ │(Process)│ │(Process)│ │(Process)│ │
│                    │  YOLO   │ │  YOLO   │ │  YOLO   │ │  YOLO   │ │
│                    └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ │
│                         └───────────┴───────────┴───────────┘      │
│                                        │                            │
│                                        ▼                            │
│                              ┌─────────────────┐                    │
│                              │  Output Queue   │                    │
│                              └────────┬────────┘                    │
│                                       │                             │
│                              ┌────────▼────────┐                    │
│                              │ Result Collector│                    │
│                              │   (Thread)      │                    │
│                              └────────┬────────┘                    │
│                                       │                             │
│                              ┌────────▼────────┐                    │
│                              │ DynamoDB Writer │                    │
│                              │ (Background)    │                    │
│                              └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Files to Create

### 1. `core/worker_pool.py` - Worker Process Management

Manages pool of inference worker processes with:
- `WorkerConfig` dataclass for per-worker settings
- `InferenceWorker.run()` static method (runs in child process)
- `WorkerPool` class for lifecycle management
- Batch-level frame distribution: `batch_id % num_workers`
- Queue sizing: `num_workers * 4` (default)
- Graceful shutdown with timeout + force terminate

### 2. `core/frame_distributor.py` - Frame Distribution

Thread-based distributor that:
- Reads frames from `FrameInputHandler`
- Auto-enables `StreamBuffer` for HLS/RTSP (via `SourceRouter.is_stream()`)
- Distributes frames using batch-level round-robin
- Sends `None` sentinel on completion for worker termination

---

## Files to Modify

### 1. `config/schemas/game.py` - Add Parallelism Settings

**Location:** `InferenceSettings` class (lines 134-144)

Add fields:
```python
class InferenceSettings(BaseModel):
    # ... existing fields ...

    # Parallelism settings
    num_workers: int = Field(default=1, ge=1)
    worker_queue_size: int = Field(default=0, ge=0)  # 0 = auto

    # Buffer settings
    enable_buffering: Optional[bool] = Field(default=None)  # None = auto
    buffer_size: int = Field(default=30, ge=1)
    buffer_mode: str = Field(default="drop_old")  # "fifo" or "drop_old"
```

### 2. `services/base_service.py` - Multi-Worker Run Mode

**Location:** `ServiceConfig` dataclass (lines 26-65)

Add fields:
```python
@dataclass
class ServiceConfig:
    # ... existing fields ...

    # Multi-worker settings
    num_workers: int = 1
    worker_queue_size: int = 0  # 0 = auto (num_workers * 4)

    # Buffer settings
    enable_buffering: Optional[bool] = None  # None = auto-detect
    buffer_size: int = 30
    buffer_mode: str = "drop_old"
```

**Location:** `BaseService` class

Add methods:
- `_run_multi_worker(num_workers, batch_size)` - orchestrates multi-worker mode
- `_collect_results(pool)` - thread that reads output queue → DB writer
- `get_model_path()` - abstract, returns model file path
- `get_model_config()` - abstract, returns model config dict
- `worker_init(config)` - static, initializes model in worker process
- `worker_process_batch(model, frames, metadata)` - static, processes batch

Modify `run()` method:
```python
def run(self):
    # ... existing setup ...

    num_workers = self.config.num_workers
    batch_size = self.config.inference_batch_size

    if num_workers > 1:
        self._run_multi_worker(num_workers, batch_size)
    elif batch_size > 1 and self.supports_batching:
        self._run_batched(batch_size)
    else:
        self._run_single()
```

### 3. `services/od_service/service.py` - Implement Worker Methods

Add implementations:
```python
def get_model_path(self) -> str:
    # Return model path from config or download

@staticmethod
def worker_init(config) -> YOLO:
    # Load YOLO model in worker process

@staticmethod
def worker_process_batch(model, frames, metadata_list) -> List[dict]:
    # Run model.predict() and format results
```

### 4. `input_handlers/frame_handler.py` - Auto-Buffer for Streams

**Location:** `__init__` method (line 96)

Change:
```python
# OLD
buffered: bool = False

# NEW
buffered: Optional[bool] = None  # None = auto-detect
```

Add auto-detection logic:
```python
if buffered is None:
    source_type = SourceRouter.detect(input_source)
    self.buffered = SourceRouter.is_stream(source_type)
    if self.buffered:
        self.buffer_mode = BufferMode.DROP_OLD  # Real-time default
```

### 5. `services/od_service/service.py` - Config Builder

**Location:** `build_od_config()` function (lines 197-284)

Pass new config params from `InferenceSettings`:
```python
return ODServiceConfig(
    # ... existing params ...
    num_workers=inference_settings.num_workers,
    worker_queue_size=inference_settings.worker_queue_size,
    enable_buffering=inference_settings.enable_buffering,
    buffer_size=inference_settings.buffer_size,
    buffer_mode=inference_settings.buffer_mode,
)
```

### 6. `main.py` - CLI Arguments

Add arguments:
```python
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--queue-size', type=int, default=0)
parser.add_argument('--buffer-mode', choices=['fifo', 'drop_old'], default='drop_old')
```

---

## Key Implementation Details

### Batch-Level Distribution Formula

```python
batch_id = frame_number // batch_size
queue_index = batch_id % num_workers
worker_queues[queue_index].put((frame, metadata))
```

This ensures all frames from the same batch go to the same worker for optimal GPU cache utilization.

### Queue Sizing

```python
queue_size = config.worker_queue_size if config.worker_queue_size > 0 else num_workers * 4
```

Default buffers 4 batches per worker - enough to hide latency without excessive memory.

### Worker Termination Sequence

1. `FrameDistributor` finishes iterating source
2. Sends `None` to each worker queue
3. Workers process remaining frames in finally block
4. Workers exit after processing `None` sentinel
5. `WorkerPool.stop()` joins with timeout, force terminates if hung
6. Result collector drains output queue
7. DB writer flushes and stops

### Auto-Buffering Logic

```python
# In FrameInputHandler.__init__
if buffered is None:
    source_type = SourceRouter.detect(input_source)
    if SourceRouter.is_stream(source_type):  # HLS or RTSP
        self.buffered = True
        self.buffer_mode = BufferMode.DROP_OLD
    else:
        self.buffered = False
```

---

## Execution Modes

| num_workers | batch_size | Mode | Description |
|-------------|------------|------|-------------|
| 1 | 1 | Single-frame | Current default, sequential |
| 1 | 8 | Batched | GPU batching, single process |
| 4 | 1 | Multi-worker | 4 processes, 1 frame each |
| 4 | 8 | Multi-worker + Batched | 4 processes, 8 frames each |

---

## Config Flow

```
YAML Config (football_v1.yaml)
    │
    ▼
InferenceSettings (game.py)
    │ num_workers, worker_queue_size, buffer_size, buffer_mode
    ▼
build_od_config() (od_service/service.py)
    │
    ▼
ODServiceConfig (extends ServiceConfig)
    │
    ▼
BaseService.run()
    │ Chooses: _run_single() / _run_batched() / _run_multi_worker()
    ▼
WorkerPool + FrameDistributor
```

---

## Verification Plan

### 1. Unit Tests
- `test_worker_pool.py`: Pool creation, start/stop, batch distribution
- `test_frame_distributor.py`: Auto-buffer detection, distribution formula

### 2. Integration Tests
Run with test video:
```bash
# Single worker (baseline)
python main.py --source test.mp4 --num-workers 1 --local

# Multi-worker
python main.py --source test.mp4 --num-workers 4 --local

# Compare outputs - should have same detections (order may differ)
```

### 3. Real-Time Stream Test
```bash
# HLS stream with auto-buffering
python main.py --source "https://example.com/stream.m3u8" --num-workers 4

# Verify:
# - "Auto-enabled buffering" log message
# - Frames distributed across workers (check logs)
# - Graceful shutdown on Ctrl+C
```

### 4. Graceful Shutdown Test
```bash
# Start long-running inference
python main.py --source long_video.mp4 --num-workers 4 &
PID=$!

# Wait 10 seconds, then send SIGTERM
sleep 10 && kill -TERM $PID

# Verify:
# - "Received signal 15" log
# - Workers finish current batch
# - Final results written to DB
# - Clean exit (no zombie processes)
```

---

## Implementation Order

1. **Config changes** (`game.py`, `base_service.py`) - Add new fields
2. **WorkerPool** (`core/worker_pool.py`) - Core multi-process logic
3. **FrameDistributor** (`core/frame_distributor.py`) - Distribution thread
4. **BaseService changes** - Add `_run_multi_worker()` and abstract methods
5. **ODService changes** - Implement `worker_init()`, `worker_process_batch()`
6. **FrameInputHandler** - Auto-buffer detection
7. **Config builders** - Pass new params through
8. **CLI args** - Add `--num-workers` etc.
9. **Tests** - Unit and integration tests

---

## Reference Files

### Current Codebase (to modify)
- `/Users/spectatr/Downloads/Inference2.0/infer_2.0/services/base_service.py`
- `/Users/spectatr/Downloads/Inference2.0/infer_2.0/services/od_service/service.py`
- `/Users/spectatr/Downloads/Inference2.0/infer_2.0/config/schemas/game.py`
- `/Users/spectatr/Downloads/Inference2.0/infer_2.0/input_handlers/frame_handler.py`
- `/Users/spectatr/Downloads/Inference2.0/infer_2.0/main.py`

### Existing Components (to integrate with)
- `/Users/spectatr/Downloads/Inference2.0/infer_2.0/core/stream_buffer.py` - StreamBuffer, BufferedFrameProvider
- `/Users/spectatr/Downloads/Inference2.0/infer_2.0/core/source_router.py` - SourceRouter.is_stream()
- `/Users/spectatr/Downloads/Inference2.0/infer_2.0/core/batch_accumulator.py` - BatchAccumulator

### Reference Implementation (old code patterns)
- `/Users/spectatr/Downloads/Inference2.0/infer_2.0/temp/old_inference_code_for_reference/src/live_stream_processing/real_time_inference.py`
  - Lines 620-626: Batch distribution formula
  - Lines 700-706: Queue creation and sizing
  - Lines 141-160: Worker batch accumulation
  - Lines 432-456: Graceful shutdown pattern
