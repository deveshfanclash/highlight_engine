# Config Migration Plan: Current → v4 Architecture

## 1. Current Config Flow Analysis

### How Config Currently Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CURRENT CONFIG FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                         CLI / Entry Point
                    ┌─────────────────────────┐
                    │ python -m orchestrator  │
                    │   --run-config          │
                    │   run_config.yaml       │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  MatchOrchestrator      │
                    │  .from_run_config()     │
                    │                         │
                    │  Reads YAML directly    │
                    │  with yaml.safe_load()  │
                    └───────────┬─────────────┘
                                │
                                │ Extracts:
                                │ - match_id, stream_url, stream_type
                                │ - deployment.local_output_dir
                                │ - deployment.hls_metadata_head_start_seconds
                                │ - deployment.enable_resume
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  _load_config()         │
                    │                         │
                    │  Returns raw dict       │
                    │  from YAML              │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ _spawn_hls_   │       │ _spawn_camera │       │ _spawn_od_    │
│ metadata()    │       │ _view()       │       │ services()    │
│               │       │               │       │               │
│ Builds CLI    │       │ Builds CLI    │       │ Builds CLI    │
│ args from     │       │ args from     │       │ args from     │
│ config dict   │       │ config dict   │       │ config dict   │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ subprocess    │       │ subprocess    │       │ subprocess    │
│ Popen()       │       │ Popen()       │       │ Popen()       │
│               │       │               │       │               │
│ python -m     │       │ python -m     │       │ python -m     │
│ services.     │       │ services.     │       │ services.     │
│ hls_metadata  │       │ camera_view   │       │ od_service    │
│ --match-id    │       │ --match-id    │       │ --match-id    │
│ --stream-url  │       │ --stream-url  │       │ --stream-url  │
│ ...           │       │ ...           │       │ ...           │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Service CLI   │       │ Service CLI   │       │ Service CLI   │
│ main()        │       │ main()        │       │ main()        │
│               │       │               │       │               │
│ argparse      │       │ argparse      │       │ argparse      │
│ → ServiceConf │       │ → ServiceConf │       │ → ODServiceCf │
└───────────────┘       └───────────────┘       └───────────────┘
```

### Key Observations

| Aspect | Current Implementation |
|--------|----------------------|
| **Config Loading** | Direct YAML read in orchestrator, no validation |
| **Config Passing** | Orchestrator → CLI args → Service argparse |
| **Service Config** | Each service has own dataclass (ODServiceConfig, etc.) |
| **Pydantic Schemas** | Exist but NOT used for runtime config loading |
| **Config Loader** | Exists but only used for MongoDB path, not unified YAML |

### Current Files Involved

```
config/
├── __init__.py              # Exports (not used in flow)
├── loader.py                # ConfigLoader (partially used)
├── class_registry.py        # Universal class definitions
├── schemas/
│   ├── __init__.py
│   ├── enums.py             # Enums (used)
│   ├── model.py             # ModelConfig (schema only)
│   ├── game.py              # GameTemplate (schema only)
│   ├── deployment.py        # DeploymentProfile (schema only)
│   ├── match.py             # MatchConfig (schema only)
│   └── output.py            # Output schemas
└── examples/
    └── run_config.yaml      # Example config (actually used)

orchestrator/
└── match_orchestrator.py    # Main entry point

services/
├── base_service.py          # ServiceRunConfig dataclass
├── od_service/
│   └── service.py           # ODServiceConfig + CLI
├── camera_view_service/
│   └── service.py           # CameraViewServiceConfig + CLI
└── hls_metadata_service/
    └── service.py           # HLSMetadataServiceConfig + CLI
```

---

## 2. New v4 Architecture

### Target Config Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEW CONFIG FLOW (v4)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                         CLI / Entry Point
                    ┌─────────────────────────┐
                    │ python -m orchestrator  │
                    │   --game football_v1    │
                    │   --match match_123     │
                    │   --source "url"        │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  NEW: ConfigResolver    │
                    │                         │
                    │  1. Load models.yaml    │
                    │  2. Load deployment.yaml│
                    │  3. Load game yaml      │
                    │  4. Apply overrides     │
                    │  5. Validate with       │
                    │     Pydantic schemas    │
                    └───────────┬─────────────┘
                                │
                                │ Returns: ResolvedConfig
                                │ (validated, merged, ready to use)
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  MatchOrchestrator      │
                    │                         │
                    │  Uses ResolvedConfig    │
                    │  directly (no dict)     │
                    └───────────┬─────────────┘
                                │
                                │ Spawns services with
                                │ typed ServiceConfig objects
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ HLSMetadata   │       │ CameraView    │       │ ODService     │
│ Service       │       │ Service       │       │               │
│               │       │               │       │               │
│ Receives      │       │ Receives      │       │ Receives      │
│ typed config  │       │ typed config  │       │ typed config  │
│ object        │       │ object        │       │ object        │
└───────────────┘       └───────────────┘       └───────────────┘
```

### New File Structure

```
config/
├── models.yaml              # NEW: Static model definitions
├── deployment.yaml          # NEW: Static deployment profiles
├── games/
│   ├── football_v1.yaml     # NEW: Game config (runtime)
│   └── cricket_v1.yaml
├── resolver.py              # NEW: Config resolution logic
├── schemas/
│   ├── v4/                  # NEW: v4 Pydantic schemas
│   │   ├── __init__.py
│   │   ├── models.py        # ModelConfig schema
│   │   ├── deployment.py    # DeploymentConfig schema
│   │   ├── game.py          # GameConfig schema
│   │   └── resolved.py      # ResolvedConfig (merged)
│   └── ... (keep old for reference)
└── examples/
    └── ... (keep for reference)

orchestrator/
└── match_orchestrator.py    # MODIFY: Use ConfigResolver

services/
├── base_service.py          # MODIFY: Update ServiceRunConfig
├── configs.py               # NEW: Centralized service configs
└── ...
```

---

## 3. Migration Steps

### Phase 1: Create New Config Files (No Code Changes)

**Step 1.1: Create `config/models.yaml`**
```yaml
# Extract from current run_config.yaml
models:
  yolo_football_v2:
    name: "YOLO Football Detector v2"
    architecture: "yolov8"
    sources:
      url: "s3://spectatr-models/yolo_football_v2.pt"
      path: "/models/yolo_football_v2.pt"
    classes:
      0: "ball"
    defaults:
      confidence: 0.5
      iou_threshold: 0.45
      max_detections: 100
      batch_size: 1
      half_precision: false
      resolution: [640, 640]
```

**Step 1.2: Create `config/deployment.yaml`**
```yaml
deployments:
  local:
    model_source: "path"
    output:
      mode: "local"
      dir: "./output"
      default_batch_size: 12
      default_flush_interval_ms: 250
    hls_head_start_seconds: 10

  prod:
    model_source: "url"
    output:
      mode: "dynamodb"
      region: "us-east-1"
      table_prefix: "prod_inference_"
      default_batch_size: 25
      default_flush_interval_ms: 100
    hls_head_start_seconds: 30
```

**Step 1.3: Create `config/games/football_v1.yaml`**
```yaml
game_id: "football_v1"
sport: "football"
deployment: "local"

input:
  hls:
    head_start_seconds: 10
    poll_interval_seconds: 5
    timeout_no_segment_seconds: 60
    resolution_preference: "_1080p.m3u8"
  mp4:
    start_frame: 0
  resume:
    enabled: false

services:
  - id: "hls_metadata"
    type: "hls_metadata"
    model: null
    enabled: true
    config:
      batch_size: 500
    output:
      table: "hls_metadata"

  - id: "camera_view"
    type: "camera_view"
    model: null
    enabled: true
    config:
      phash_threshold: 20
      histogram_threshold: 0.9
      min_frame_gap: 25
      resolution_scale: 0.5
      frame_skip: 1
    output:
      table: "camera_view"
      batch_size: 12

  - id: "object_detection:yolo_football_v2"
    type: "object_detection"
    model: "yolo_football_v2"
    enabled: true
    config:
      device: "cpu"
      frame_skip: 1
      target_width: 1280
      target_height: 720
      classes_to_predict: [0]
    output:
      table: "detections"
      batch_size: 12

overrides:
  models:
    yolo_football_v2:
      confidence: 0.6
  deployment:
    output:
      dir: "./test_output"
```

---

### Phase 2: Create Config Resolver

**Step 2.1: Create `config/schemas/v4/models.py`**
```python
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ModelSources(BaseModel):
    url: str = ""
    path: str = ""

class ModelDefaults(BaseModel):
    confidence: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    batch_size: int = 1
    half_precision: bool = False
    resolution: List[int] = Field(default=[640, 640])

class ModelConfig(BaseModel):
    name: str = ""
    architecture: str = "yolov8"
    output_format: str = "bbox"
    sources: ModelSources = Field(default_factory=ModelSources)
    classes: Dict[int, str] = Field(default_factory=dict)
    defaults: ModelDefaults = Field(default_factory=ModelDefaults)

class ModelsConfig(BaseModel):
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
```

**Step 2.2: Create `config/schemas/v4/deployment.py`**
```python
from typing import Optional
from pydantic import BaseModel, Field

class OutputConfig(BaseModel):
    mode: str = "local"  # "local" or "dynamodb"
    dir: Optional[str] = "./output"
    region: str = "us-east-1"
    table_prefix: str = ""
    default_batch_size: int = 12
    default_flush_interval_ms: int = 250

class DeploymentProfile(BaseModel):
    model_source: str = "path"  # "url" or "path"
    output: OutputConfig = Field(default_factory=OutputConfig)
    hls_head_start_seconds: int = 10

class DeploymentsConfig(BaseModel):
    deployments: Dict[str, DeploymentProfile] = Field(default_factory=dict)
```

**Step 2.3: Create `config/schemas/v4/game.py`**
```python
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ServiceOutputConfig(BaseModel):
    table: str = "inference_results"
    batch_size: Optional[int] = None  # None = use deployment default

class ServiceConfig(BaseModel):
    id: str
    type: str
    model: Optional[str] = None
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)
    output: ServiceOutputConfig = Field(default_factory=ServiceOutputConfig)

class InputHLSConfig(BaseModel):
    head_start_seconds: int = 10
    poll_interval_seconds: int = 5
    timeout_no_segment_seconds: int = 60
    resolution_preference: str = "_480p.m3u8"

class InputMP4Config(BaseModel):
    start_frame: int = 0

class ResumeConfig(BaseModel):
    enabled: bool = False
    start_frame: int = 0
    start_segment: int = 1

class InputConfig(BaseModel):
    hls: InputHLSConfig = Field(default_factory=InputHLSConfig)
    mp4: InputMP4Config = Field(default_factory=InputMP4Config)
    resume: ResumeConfig = Field(default_factory=ResumeConfig)

class OverridesConfig(BaseModel):
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    deployment: Dict[str, Any] = Field(default_factory=dict)

class GameConfig(BaseModel):
    game_id: str
    sport: str = ""
    description: str = ""
    deployment: str = "local"
    input: InputConfig = Field(default_factory=InputConfig)
    services: List[ServiceConfig] = Field(default_factory=list)
    overrides: OverridesConfig = Field(default_factory=OverridesConfig)
```

**Step 2.4: Create `config/resolver.py`**
```python
"""
Config Resolver - Loads and merges configs from v4 YAML files
"""
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from config.schemas.v4.models import ModelsConfig, ModelConfig
from config.schemas.v4.deployment import DeploymentsConfig, DeploymentProfile
from config.schemas.v4.game import GameConfig

@dataclass
class ResolvedServiceConfig:
    """Fully resolved config for a single service"""
    id: str
    type: str
    model_id: Optional[str]
    enabled: bool

    # Model settings (resolved from model + overrides)
    model_path: Optional[str]
    model_architecture: str
    confidence: float
    iou_threshold: float
    max_detections: int
    batch_size: int
    half_precision: bool
    classes_to_predict: List[int]
    class_mapping: Dict[int, str]

    # Service settings
    device: str
    frame_skip: int
    target_width: Optional[int]
    target_height: Optional[int]

    # Service-specific (camera_view, hls_metadata, etc.)
    service_params: Dict[str, Any]

    # Output settings
    db_table_name: str
    db_batch_size: int
    db_flush_interval_ms: int
    local_output_dir: Optional[str]

@dataclass
class ResolvedConfig:
    """Fully resolved and validated config"""
    # Runtime
    match_id: str
    source: str
    input_type: str  # "hls" or "mp4"

    # Game
    game_id: str
    sport: str

    # Deployment
    deployment_id: str
    local_output_dir: Optional[str]
    hls_head_start_seconds: int

    # Resume
    resume_enabled: bool
    start_frame: int
    start_segment: int

    # Services (fully resolved)
    services: List[ResolvedServiceConfig]

class ConfigResolver:
    """Loads and resolves v4 configs"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._models: Optional[ModelsConfig] = None
        self._deployments: Optional[DeploymentsConfig] = None

    def resolve(
        self,
        game_id: str,
        match_id: str,
        source: str,
        input_type: Optional[str] = None,
        deployment_override: Optional[str] = None,
        resume: bool = False,
    ) -> ResolvedConfig:
        """
        Load and resolve all configs for a run.

        Args:
            game_id: Game config ID (e.g., "football_v1")
            match_id: Unique match identifier
            source: Stream URL or file path
            input_type: "hls" or "mp4" (auto-detected if None)
            deployment_override: Override deployment from game config
            resume: Enable resume from last position

        Returns:
            Fully resolved ResolvedConfig
        """
        # Auto-detect input type
        if input_type is None:
            input_type = self._detect_input_type(source)

        # Load base configs
        models = self._load_models()
        deployments = self._load_deployments()
        game = self._load_game(game_id)

        # Determine deployment
        deployment_id = deployment_override or game.deployment
        deployment = deployments.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment not found: {deployment_id}")

        # Apply deployment overrides from game config
        deployment = self._apply_deployment_overrides(deployment, game.overrides.deployment)

        # Resolve each service
        resolved_services = []
        for svc in game.services:
            if not svc.enabled:
                continue
            resolved = self._resolve_service(svc, models, deployment, game)
            resolved_services.append(resolved)

        # Build final config
        return ResolvedConfig(
            match_id=match_id,
            source=source,
            input_type=input_type,
            game_id=game.game_id,
            sport=game.sport,
            deployment_id=deployment_id,
            local_output_dir=deployment.output.dir if deployment.output.mode == "local" else None,
            hls_head_start_seconds=deployment.hls_head_start_seconds,
            resume_enabled=resume or game.input.resume.enabled,
            start_frame=game.input.resume.start_frame,
            start_segment=game.input.resume.start_segment,
            services=resolved_services,
        )

    def _detect_input_type(self, source: str) -> str:
        if source.endswith(".m3u8") or "m3u8" in source:
            return "hls"
        elif source.endswith(".mp4"):
            return "mp4"
        else:
            return "hls"  # Default

    def _load_models(self) -> ModelsConfig:
        if self._models is None:
            path = self.config_dir / "models.yaml"
            with open(path) as f:
                data = yaml.safe_load(f)
            self._models = ModelsConfig(**data)
        return self._models

    def _load_deployments(self) -> DeploymentsConfig:
        if self._deployments is None:
            path = self.config_dir / "deployment.yaml"
            with open(path) as f:
                data = yaml.safe_load(f)
            self._deployments = DeploymentsConfig(**data)
        return self._deployments

    def _load_game(self, game_id: str) -> GameConfig:
        path = self.config_dir / "games" / f"{game_id}.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        return GameConfig(**data)

    def _resolve_service(
        self,
        svc: ServiceConfig,
        models: ModelsConfig,
        deployment: DeploymentProfile,
        game: GameConfig,
    ) -> ResolvedServiceConfig:
        """Resolve a single service config"""
        # ... implementation
        pass
```

---

### Phase 3: Update Orchestrator

**Step 3.1: Modify `orchestrator/match_orchestrator.py`**

```python
# Add new method to MatchOrchestrator

@classmethod
def from_v4_config(
    cls,
    game_id: str,
    match_id: str,
    source: str,
    input_type: Optional[str] = None,
    deployment: Optional[str] = None,
    resume: bool = False,
    config_dir: str = "config",
) -> "MatchOrchestrator":
    """
    Create orchestrator from v4 config files.

    Args:
        game_id: Game config ID (e.g., "football_v1")
        match_id: Unique match identifier
        source: Stream URL or file path
        input_type: "hls" or "mp4" (auto-detected if None)
        deployment: Override deployment profile
        resume: Enable resume
        config_dir: Config directory path
    """
    from config.resolver import ConfigResolver

    resolver = ConfigResolver(config_dir)
    config = resolver.resolve(
        game_id=game_id,
        match_id=match_id,
        source=source,
        input_type=input_type,
        deployment_override=deployment,
        resume=resume,
    )

    instance = cls(
        match_id=config.match_id,
        stream_url=config.source,
        stream_type=config.input_type,
        local_output_dir=config.local_output_dir,
        enable_resume=config.resume_enabled,
        hls_head_start_seconds=config.hls_head_start_seconds,
    )
    instance._resolved_config = config
    return instance
```

**Step 3.2: Update CLI in `orchestrator/match_orchestrator.py`**

```python
def main():
    parser = argparse.ArgumentParser()

    # NEW v4 args
    parser.add_argument("--game", help="Game config ID")
    parser.add_argument("--match", help="Match identifier")
    parser.add_argument("--source", help="Stream URL or file path")
    parser.add_argument("--input-type", choices=["hls", "mp4"])
    parser.add_argument("--deployment", help="Override deployment profile")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config-dir", default="config")

    # Legacy args (keep for backward compat during transition)
    parser.add_argument("--run-config", help="[LEGACY] Path to run config YAML")

    args = parser.parse_args()

    if args.game:
        # NEW v4 path
        orchestrator = MatchOrchestrator.from_v4_config(
            game_id=args.game,
            match_id=args.match,
            source=args.source,
            input_type=args.input_type,
            deployment=args.deployment,
            resume=args.resume,
            config_dir=args.config_dir,
        )
    elif args.run_config:
        # Legacy path
        orchestrator = MatchOrchestrator.from_run_config(args.run_config)
    else:
        parser.error("Either --game or --run-config required")

    orchestrator.start()
    orchestrator.wait()
```

---

### Phase 4: Update Service Spawning

**Step 4.1: Update `_spawn_*` methods to use ResolvedServiceConfig**

```python
def _spawn_service(self, service_config: ResolvedServiceConfig):
    """Spawn a service using resolved config"""

    if service_config.type == "hls_metadata":
        cmd = self._build_hls_command(service_config)
    elif service_config.type == "camera_view":
        cmd = self._build_camera_view_command(service_config)
    elif service_config.type == "object_detection":
        cmd = self._build_od_command(service_config)
    else:
        logger.warning(f"Unknown service type: {service_config.type}")
        return

    process = subprocess.Popen(cmd)
    self._processes.append(ServiceProcess(
        service_id=service_config.id,
        service_type=service_config.type,
        process=process,
        device=service_config.device,
        started_at=datetime.utcnow()
    ))

def _build_od_command(self, cfg: ResolvedServiceConfig) -> List[str]:
    """Build OD service command from resolved config"""
    cmd = [
        sys.executable, "-m", "services.od_service",
        "--match-id", self._resolved_config.match_id,
        "--stream-url", self._resolved_config.source,
        "--input-type", self._resolved_config.input_type,
        "--model-id", cfg.model_id,
        "--model-path", cfg.model_path,
        "--device", cfg.device,
        "--confidence", str(cfg.confidence),
        "--start-frame", str(self._resume_position.get("frame_number", 0)),
        "--start-segment", str(self._resume_position.get("segment_number", 1)),
    ]

    if cfg.target_width:
        cmd.extend(["--width", str(cfg.target_width)])
    if cfg.target_height:
        cmd.extend(["--height", str(cfg.target_height)])
    if cfg.classes_to_predict:
        cmd.extend(["--classes", ",".join(str(c) for c in cfg.classes_to_predict)])
    if cfg.class_mapping:
        cmd.extend(["--class-mapping", json.dumps(cfg.class_mapping)])
    if cfg.local_output_dir:
        cmd.extend(["--local-output", cfg.local_output_dir])

    cmd.extend(["--db-table", cfg.db_table_name])

    return cmd
```

---

## 4. Testing Strategy

### Step-by-Step Testing

```bash
# Phase 1: Test config loading (no service changes)
python -c "
from config.resolver import ConfigResolver
resolver = ConfigResolver('config')
config = resolver.resolve('football_v1', 'test_match', 'https://example.m3u8')
print(config)
"

# Phase 2: Test orchestrator with v4 config (services still use CLI)
python -m orchestrator.match_orchestrator \
  --game football_v1 \
  --match test_001 \
  --source "https://example.com/stream.m3u8" \
  --deployment local

# Phase 3: Test individual services
python -m services.od_service \
  --match-id test \
  --stream-url "test.mp4" \
  --model-id yolo_v2 \
  --model-path /models/yolo.pt \
  --local-output ./output
```

---

## 5. Summary: Files to Create/Modify

### New Files to Create

| File | Purpose |
|------|---------|
| `config/models.yaml` | Static model definitions |
| `config/deployment.yaml` | Static deployment profiles |
| `config/games/football_v1.yaml` | Football game config |
| `config/resolver.py` | Config resolution logic |
| `config/schemas/v4/__init__.py` | v4 schema exports |
| `config/schemas/v4/models.py` | Model schema |
| `config/schemas/v4/deployment.py` | Deployment schema |
| `config/schemas/v4/game.py` | Game schema |
| `config/schemas/v4/resolved.py` | Resolved config |

### Files to Modify

| File | Changes |
|------|---------|
| `orchestrator/match_orchestrator.py` | Add `from_v4_config()`, update CLI |
| `services/base_service.py` | No changes (CLI interface unchanged) |
| `services/od_service/service.py` | No changes (CLI interface unchanged) |
| `services/camera_view_service/service.py` | No changes |
| `services/hls_metadata_service/service.py` | No changes |

### Files to Keep (Reference)

| File | Status |
|------|--------|
| `config/loader.py` | Keep for MongoDB path |
| `config/schemas/*.py` | Keep for reference |
| `config/examples/*.yaml` | Keep as examples |

---

## 6. Questions to Clarify

Before starting implementation:

1. **Model path resolution**: When `deployment.model_source = "path"`, should we:
   - Use `model.sources.path` directly?
   - Or allow override in `game.overrides.models[id].source`?

2. **Service CLI vs direct config**: Should services eventually accept config JSON instead of CLI args?
   - Current: CLI args (works, but verbose)
   - Future: `--config config.json` or environment variable

3. **Resume position**: Should `start_frame`/`start_segment` be:
   - Set in game yaml (static)?
   - Or always queried from HLS metadata at runtime?

4. **Validation**: How strict should schema validation be?
   - Fail on unknown keys?
   - Warn on unknown keys?

Let me know your answers and we can start implementation.
