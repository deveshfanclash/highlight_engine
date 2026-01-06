#!/bin/bash
# Local testing script for inference services
# Usage: ./test_services_local.sh /path/to/video.mp4

set -e

VIDEO_PATH="${1:-/tmp/test_video.mp4}"
OUTPUT_DIR="/tmp/inference_output"
MATCH_ID="test_$(date +%s)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Inference Services Local Test ===${NC}"
echo "Video: $VIDEO_PATH"
echo "Output: $OUTPUT_DIR"
echo "Match ID: $MATCH_ID"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if video exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Video file not found: $VIDEO_PATH"
    echo "Downloading sample video..."
    curl -L -o "$VIDEO_PATH" "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4" 2>/dev/null || {
        echo "Download failed. Please provide a valid video path."
        exit 1
    }
fi

# Test 1: Camera View Service
echo -e "\n${GREEN}[1/2] Testing Camera View Service...${NC}"
timeout 30 python -m services.camera_view_service \
    --match-id "$MATCH_ID" \
    --stream-url "$VIDEO_PATH" \
    --local-output "$OUTPUT_DIR" \
    --width 640 \
    --height 360 \
    --min-frame-gap 10 \
    2>&1 | head -30 || echo "Camera view test completed (or timed out after 30s)"

# Test 2: Object Detection Service (if model available)
if [ -f "./yolov8n.pt" ] || [ -f "$HOME/.cache/ultralytics/yolov8n.pt" ]; then
    MODEL_PATH="./yolov8n.pt"
    [ -f "$HOME/.cache/ultralytics/yolov8n.pt" ] && MODEL_PATH="$HOME/.cache/ultralytics/yolov8n.pt"

    echo -e "\n${GREEN}[2/2] Testing Object Detection Service...${NC}"
    timeout 30 python -m services.od_service \
        --match-id "$MATCH_ID" \
        --stream-url "$VIDEO_PATH" \
        --model-id yolov8n \
        --model-path "$MODEL_PATH" \
        --device cpu \
        --confidence 0.3 \
        --local-output "$OUTPUT_DIR" \
        --class-mapping '{"0": "PERSON"}' \
        2>&1 | head -30 || echo "OD test completed (or timed out after 30s)"
else
    echo -e "\n${GREEN}[2/2] Skipping OD Service - No model found${NC}"
    echo "To test OD, first run: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\""
fi

# Show results
echo -e "\n${BLUE}=== Results ===${NC}"
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No output files generated"

echo -e "\n${GREEN}Done!${NC}"
