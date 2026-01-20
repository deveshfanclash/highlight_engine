# Config Design v4 - Complete Implementation

## Overview

Based on analysis of current codebase implementation, this design includes ALL parameters currently in use.

---

## File Structure

```
config/
├── models.yaml              # Model definitions (static)
├── deployment.yaml          # Environment configs (static)
└── games/
    └── football_v1.yaml     # Game config (runtime - ONLY file to edit)
```

---

## 1. Models YAML (Static)

**File:** `config/models.yaml`

```yaml
# =============================================================================
# MODEL REGISTRY
# =============================================================================

models:
  yolo_football_v2:
    name: "YOLO Football Detector v2"
    architecture: "yolov8"              # yolov8, yolov11, yolov12, rtdetr, custom
    output_format: "bbox"               # bbox, mask, keypoints

    # Model sources (deployment decides which to use)
    sources:
      url: "s3://spectatr-models/yolo_football_v2.pt"
      path: "/models/yolo_football_v2.pt"

    # Class definitions
    classes:
      0: "ball"
      1: "person"
      2: "goalkeeper"
      3: "referee"

    # Default inference parameters
    defaults:
      confidence: 0.5
      iou_threshold: 0.45
      max_detections: 100
      batch_size: 1
      half_precision: false
      resolution: [640, 640]            # Model input size

  yolo_ball_tracker_v1:
    name: "Ball Tracker"
    architecture: "yolov8"
    output_format: "bbox"
    sources:
      url: "s3://spectatr-models/ball_tracker_v1.pt"
      path: "/models/ball_tracker_v1.pt"
    classes:
      0: "ball"
    defaults:
      confidence: 0.3
      iou_threshold: 0.45
      max_detections: 50
      batch_size: 1
      half_precision: false
      resolution: [640, 640]
```

---

## 2. Deployment YAML (Static)

**File:** `config/deployment.yaml`

```yaml
# =============================================================================
# DEPLOYMENT PROFILES
# =============================================================================

deployments:
  # ---------------------------------------------------------------------------
  # Production
  # ---------------------------------------------------------------------------
  prod:
    model_source: "url"                 # Use sources.url from model

    output:
      # mode: "dynamodb"
      region: "us-east-1"
      table_prefix: "prod_inference_"
      default_batch_size: 25
      default_flush_interval_ms: 100

    # Orchestrator settings
    hls_head_start_seconds: 30          # Wait before starting other services

    # AWS (credentials from environment variables)
    aws:
      access_key_env: "AWS_ACCESS_KEY"
      secret_key_env: "AWS_SECRET_KEY"

    kafka:
      enabled: true
      bootstrap_servers: "kafka-prod.internal:9092"

    logging:
      level: "INFO"

  # ---------------------------------------------------------------------------
  # Development
  # ---------------------------------------------------------------------------
  dev:
    model_source: "url"

    output:
      mode: "dynamodb"
      region: "us-east-2"
      table_prefix: "dev_inference_"
      default_batch_size: 20
      default_flush_interval_ms: 200

    hls_head_start_seconds: 15

    kafka:
      enabled: false

    logging:
      level: "DEBUG"

  # ---------------------------------------------------------------------------
  # Local
  # ---------------------------------------------------------------------------
  local:
    model_source: "path"                # Use sources.path from model

    output:
      mode: "local"
      dir: "./output"
      default_batch_size: 12
      default_flush_interval_ms: 250

    hls_head_start_seconds: 10

    kafka:
      enabled: false

    logging:
      level: "DEBUG"
```

---

## 3. Game YAML (Runtime Config - Single Source of Truth)

**File:** `config/games/football_v1.yaml`

```yaml
# =============================================================================
# GAME CONFIGURATION - FOOTBALL v1
# =============================================================================
# ONLY file to modify at runtime. All other configs are static defaults.

game_id: "football_v1"
sport: "football"
description: "Football/Soccer inference pipeline"

# =============================================================================
# DEPLOYMENT SELECTION
# =============================================================================
deployment: "local"                     # Options: prod, dev, local

# =============================================================================
# INPUT CONFIGURATION
# =============================================================================
# Input type auto-detected from --source CLI arg (.m3u8 → hls, .mp4 → mp4)

input:
  # HLS-specific settings
  hls:
    head_start_seconds: 10              # Override deployment default
    poll_interval_seconds: 5
    timeout_no_segment_seconds: 60
    resolution_preference: "_1080p.m3u8"

  # MP4-specific settings
  mp4:
    start_frame: 0

  # Resume settings
  resume:
    enabled: false                      # Enable resume from last position
    # If enabled, these are auto-populated from HLS metadata:
    # start_frame: 0
    # start_segment: 1

# =============================================================================
# SERVICES
# =============================================================================
# Execution order: top to bottom. Service ID format: {type}:{model} or {type}

services:
  # ---------------------------------------------------------------------------
  # HLS Metadata (runs first - required for resume)
  # ---------------------------------------------------------------------------
  - id: "hls_metadata"
    type: "hls_metadata"
    model: null
    enabled: true
    config:
      # Inherits from input.hls section
      batch_size: 500                   # HLS has larger batches
    output:
      table: "hls_metadata"

  # ---------------------------------------------------------------------------
  # Camera View Detection
  # ---------------------------------------------------------------------------
  - id: "camera_view"
    type: "camera_view"
    model: null
    enabled: true
    config:
      phash_threshold: 20
      histogram_threshold: 0.9
      min_frame_gap: 25
      resolution_scale: 0.5             # Scale factor for processing
      frame_skip: 1
      # Explicit resolution (optional - computed from resolution_scale if not set)
      # target_width: 640
      # target_height: 360
    output:
      table: "camera_view"
      batch_size: 12

  # ---------------------------------------------------------------------------
  # Object Detection - Main
  # ---------------------------------------------------------------------------
  - id: "object_detection:yolo_football_v2"
    type: "object_detection"
    model: "yolo_football_v2"           # References models.yaml
    enabled: true
    config:
      device: "cpu"                     # cpu, cuda:0, cuda:1, mps
      frame_skip: 1
      target_width: 1280                # Processing resolution
      target_height: 720
      classes_to_predict: [0, 1]        # Model class IDs to detect (empty = all)
      # Model param overrides (optional)
      # confidence: 0.6                 # Override model default
      # iou_threshold: 0.5
      # batch_size: 4
      # half_precision: true
    output:
      table: "detections"
      batch_size: 12

  # ---------------------------------------------------------------------------
  # Object Detection - Ball Tracker (optional)
  # ---------------------------------------------------------------------------
  - id: "object_detection:yolo_ball_tracker_v1"
    type: "object_detection"
    model: "yolo_ball_tracker_v1"
    enabled: false
    config:
      device: "cpu"
      frame_skip: 2
      target_width: 1280
      target_height: 720
      classes_to_predict: [0]           # Ball only
    output:
      table: "ball_detections"
      batch_size: 12

# =============================================================================
# OVERRIDES (Applied on top of defaults)
# =============================================================================

overrides:
  # ---------------------------------------------------------------------------
  # Model Overrides
  # ---------------------------------------------------------------------------
  models:
    yolo_football_v2:
      confidence: 0.6                   # Override default 0.5
      # source: "/custom/path/model.pt" # Override model path

  # ---------------------------------------------------------------------------
  # Deployment Overrides
  # ---------------------------------------------------------------------------
  deployment:
    output:
      dir: "./test_output"
      # table_prefix: "test_"

# =============================================================================
# UNIVERSAL CLASS MAPPING (Optional)
# =============================================================================
# Maps model classes to display names for this game

class_mapping:
  BALL:
    display_name: "Ball"
    color: "#FFFF00"
  PERSON:
    display_name: "Player"
    color: "#00FF00"
  GOALKEEPER:
    display_name: "Goalkeeper"
    color: "#0000FF"
  REFEREE:
    display_name: "Referee"
    color: "#FF0000"
```

---

## 4. CLI Interface

```bash
# =============================================================================
# ORCHESTRATOR CLI
# =============================================================================

# Basic usage (input type auto-detected)
python -m orchestrator.main \
  --game football_v1 \
  --match match_12345 \
  --source "https://cdn.example.com/stream.m3u8"

# With explicit options
python -m orchestrator.main \
  --game football_v1 \
  --match match_12345 \
  --source "/videos/match.mp4" \
  --input-type mp4 \
  --deployment local \
  --resume \
  --dry-run
```

**CLI Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--game` | Yes | - | Game config ID |
| `--match` | Yes | - | Unique match identifier |
| `--source` | Yes | - | Stream URL or file path |
| `--input-type` | No | auto | `hls` or `mp4` |
| `--deployment` | No | from game | Override deployment profile |
| `--resume` | No | false | Resume from last position |
| `--dry-run` | No | false | Validate without executing |

---

## 5. Complete Parameter Reference

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `architecture` | str | "yolov8" | Model architecture |
| `output_format` | str | "bbox" | Output type |
| `sources.url` | str | - | Remote model URL |
| `sources.path` | str | - | Local model path |
| `classes` | dict | - | class_id → class_name |
| `defaults.confidence` | float | 0.5 | Confidence threshold |
| `defaults.iou_threshold` | float | 0.45 | NMS IoU threshold |
| `defaults.max_detections` | int | 100 | Max detections per frame |
| `defaults.batch_size` | int | 1 | Inference batch size |
| `defaults.half_precision` | bool | false | FP16 inference |
| `defaults.resolution` | list | [640,640] | Model input size |

### Deployment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_source` | str | "url" | "url" or "path" |
| `output.mode` | str | - | "dynamodb" or "local" |
| `output.region` | str | "us-east-1" | AWS region |
| `output.dir` | str | "./output" | Local output directory |
| `output.table_prefix` | str | "" | Table name prefix |
| `output.default_batch_size` | int | 25 | Default batch size |
| `output.default_flush_interval_ms` | int | 100 | Flush interval |
| `hls_head_start_seconds` | int | 30 | HLS service head start |

### Service Parameters

#### HLS Metadata
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `poll_interval_seconds` | int | 5 | Playlist poll interval |
| `timeout_no_segment_seconds` | int | 60 | Timeout for no new segments |
| `resolution_preference` | str | "_480p.m3u8" | Preferred variant |
| `batch_size` | int | 500 | Write batch size |

#### Camera View
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phash_threshold` | int | 20 | pHash difference threshold |
| `histogram_threshold` | float | 0.9 | Histogram correlation threshold |
| `min_frame_gap` | int | 25 | Min frames between detections |
| `resolution_scale` | float | 0.5 | Scale factor for processing |
| `frame_skip` | int | 1 | Process every Nth frame |
| `target_width` | int | computed | Processing width |
| `target_height` | int | computed | Processing height |

#### Object Detection
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | "cpu" | Inference device |
| `frame_skip` | int | 1 | Process every Nth frame |
| `target_width` | int | 1280 | Processing width |
| `target_height` | int | 720 | Processing height |
| `classes_to_predict` | list | [] | Class IDs to detect (empty=all) |
| `confidence` | float | from model | Override confidence |
| `iou_threshold` | float | from model | Override IoU |
| `batch_size` | int | from model | Override batch size |
| `half_precision` | bool | from model | Override precision |

---

## 6. Config Resolution Flow

```
CLI: --game football_v1 --match m123 --source "url" --resume
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STEP 1: Load Base Configs                         │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
  ┌───────────┐        ┌───────────┐        ┌───────────┐
  │ models    │        │deployment │        │  game     │
  │ .yaml     │        │ .yaml     │        │  .yaml    │
  └─────┬─────┘        └─────┬─────┘        └─────┬─────┘
        │                    │                    │
        │                    │     deployment:    │
        │                    │◄────"local"────────┤
        │                    │                    │
        │   model: "yolo_v2" │                    │
        │◄───────────────────┼────────────────────┤
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STEP 2: Apply Overrides                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
   Priority (low → high):     │
   1. Model defaults          │
   2. Deployment defaults     │
   3. Game yaml overrides     │
   4. CLI args                │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STEP 3: Resolve Resume                               │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
   If --resume flag:          │
   1. Query HLS metadata      │
   2. Get last segment/frame  │
   3. Set start_segment,      │
      start_frame             │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STEP 4: Build Service Configs                        │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
   For each enabled service:  │
   1. Merge base + overrides  │
   2. Resolve model path      │
   3. Apply device setting    │
   4. Build ServiceRunConfig  │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STEP 5: Execute                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
   1. Start HLS Metadata      │
   2. Wait hls_head_start     │
   3. Start remaining services│
   4. Monitor and collect     │
```

---

## 7. Migration from Current Config

### What Changes

| Current | v4 | Notes |
|---------|-----|-------|
| `model_url` / `model_path` separate | `sources.url` / `sources.path` | Grouped |
| `default_params.confidence_threshold` | `defaults.confidence` | Shorter |
| `class_filter` in ModelAssignment | `classes_to_predict` in service | Clearer location |
| `processing_resolution` global | `target_width/height` per service | More flexible |
| `hls_metadata_head_start_seconds` in deployment | `hls_head_start_seconds` | Shortened |
| No `device` in config | `device` per service | Added |
| No explicit resume | `input.resume.enabled` | Added |

### What Stays Same

- 4-tier concept (model, deployment, game, runtime)
- Override hierarchy
- Service-based architecture
- YAML format

---

## 8. Quick Reference

### What Goes Where?

| Config Type | File | When to Change |
|-------------|------|----------------|
| Model architecture, classes | `models.yaml` | New model version |
| AWS region, credentials | `deployment.yaml` | New environment |
| Which services run | `game.yaml` | Per-game tuning |
| Confidence thresholds | `game.yaml` overrides | Runtime tuning |
| Input source, match ID | CLI args | Every run |

### Common Operations

```bash
# Switch to production
# Edit game.yaml: deployment: "prod"

# Increase confidence
# Edit game.yaml: overrides.models.yolo_football_v2.confidence: 0.7

# Enable ball tracker
# Edit game.yaml: services[ball_tracker].enabled: true

# Use GPU
# Edit game.yaml: services[od].config.device: "cuda:0"

# Resume failed match
python -m orchestrator.main --game football_v1 --match match_123 --source "..." --resume
```
