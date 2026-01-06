# Adding a New Game/Sport

## Quick Start (5 minutes)

1. **Copy the template:**
   ```bash
   cp config/examples/new_game_template.yaml config/examples/volleyball_config.yaml
   ```

2. **Edit your config:**
   - Set `game_id`, `game_name`, `game_category`
   - Add your model(s) with S3 URL
   - Map model classes to universal classes
   - Enable/disable services

3. **Test your config:**
   ```bash
   python -m config.loader config/examples/volleyball_config.yaml
   ```

4. **Run inference:**
   ```bash
   # Local testing (no AWS)
   python -m orchestrator.match_orchestrator \
     --match-id test_match_001 \
     --config config/examples/volleyball_config.yaml \
     --stream-url /path/to/video.mp4 \
     --local-output /tmp/test_output

   # Production (DynamoDB + S3)
   python -m orchestrator.match_orchestrator \
     --match-id match_123 \
     --config config/examples/volleyball_config.yaml \
     --stream-url "https://cdn.example.com/match.m3u8"
   ```

## What You Get

- **Object Detection**: Bounding boxes stored in DynamoDB
- **Camera View Detection**: Cut points for scene segmentation
- **Resume Support**: Automatic resume from last processed frame
- **Multi-GPU**: Each model can run on different GPU
- **Local Testing**: No AWS needed for development

## Example Configs

| File | Description |
|------|-------------|
| `football_config.yaml` | Single model, simple setup |
| `cricket_config.yaml` | Multi-model (3 models on 3 GPUs) |
| `new_game_template.yaml` | Template with all options documented |

## Universal Classes

See `config/class_registry.py` for available universal class names:

| ID Range | Category | Examples |
|----------|----------|----------|
| 1-99 | Common | PERSON, BALL, GOAL, HEAD |
| 100-199 | Football | GOALKEEPER, REFEREE |
| 200-299 | Cricket | BATSMAN, BOWLER, WICKET |
| 300-399 | Tennis | RACKET, NET, COURT_LINE |

## Adding Custom Logic

For custom preprocessing or post-processing:

1. **Custom Model**: Extend `models/base_model.py`
2. **Custom Service**: Extend `services/base_service.py`
3. **Custom Output**: Add S3 storage for masks/keypoints

See `services/od_service/service.py` for a complete example.
