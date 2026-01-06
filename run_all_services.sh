#!/bin/bash
# Run all inference services in parallel
# Usage: ./run_all_services.sh /path/to/video.mp4 [match_id]

set -e

VIDEO="${1:-/Users/spectatr/Downloads/1_clip1.mp4}"
MATCH_ID="${2:-match_$(date +%s)}"
OUTPUT="/tmp/inference_output"
CUSTOM_MODEL="/Users/spectatr/Downloads/best_263_nano_1920_bph_v3.pt"

echo "=========================================="
echo "Running Inference Services"
echo "=========================================="
echo "Video: $VIDEO"
echo "Match ID: $MATCH_ID"
echo "Output: $OUTPUT"
echo "=========================================="

mkdir -p $OUTPUT

# Check video exists
if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Video not found: $VIDEO"
    exit 1
fi

# Start Camera View
echo "[1/3] Starting Camera View Service..."
python -m services.camera_view_service \
  --match-id $MATCH_ID \
  --stream-url $VIDEO \
  --local-output $OUTPUT \
  --width 640 --height 360 \
  > /tmp/log_camera_view.txt 2>&1 &
PID_CV=$!

# Start OD - Person (default YOLO)
echo "[2/3] Starting OD Service (Person)..."
python -m services.od_service \
  --match-id $MATCH_ID \
  --stream-url $VIDEO \
  --model-id yolo_person \
  --model-path ./yolov8n.pt \
  --device cpu \
  --confidence 0.4 \
  --classes "0" \
  --class-mapping '{"0": "PERSON"}' \
  --local-output $OUTPUT \
  > /tmp/log_od_person.txt 2>&1 &
PID_OD1=$!

# Start OD - Custom Model
echo "[3/3] Starting OD Service (Custom BPH v3)..."
python -m services.od_service \
  --match-id $MATCH_ID \
  --stream-url $VIDEO \
  --model-id bph_v3 \
  --model-path $CUSTOM_MODEL \
  --device cpu \
  --confidence 0.4 \
  --local-output $OUTPUT \
  > /tmp/log_od_custom.txt 2>&1 &
PID_OD2=$!

echo ""
echo "Services running:"
echo "  Camera View: PID $PID_CV"
echo "  OD Person:   PID $PID_OD1"
echo "  OD Custom:   PID $PID_OD2"
echo ""
echo "Monitor logs:"
echo "  tail -f /tmp/log_camera_view.txt"
echo "  tail -f /tmp/log_od_person.txt"
echo "  tail -f /tmp/log_od_custom.txt"
echo ""

# Wait for all to complete
echo "Waiting for services to complete..."
wait $PID_CV && echo "✓ Camera View completed" || echo "✗ Camera View failed"
wait $PID_OD1 && echo "✓ OD Person completed" || echo "✗ OD Person failed"
wait $PID_OD2 && echo "✓ OD Custom completed" || echo "✗ OD Custom failed"

echo ""
echo "=========================================="
echo "Results in: $OUTPUT"
echo "=========================================="
ls -la $OUTPUT/${MATCH_ID}*.jsonl 2>/dev/null || echo "No output files found"
