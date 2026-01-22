#!/usr/bin/env python3
"""
Integration Test for New Components

Tests:
1. StreamBuffer - backpressure and buffering
2. ModelDownloader - caching and source detection
3. Typed Results - DetectionResults, filtering, serialization
4. YOLOModel - returns DetectionResults
5. Full pipeline with BufferedFrameProvider

Run:
    python test_integration.py
"""

import sys
import time
import tempfile
import json
from pathlib import Path
from threading import Thread

import numpy as np


def test_stream_buffer():
    """Test StreamBuffer component."""
    print("\n" + "="*60)
    print("TEST: StreamBuffer")
    print("="*60)

    from core.stream_buffer import StreamBuffer, BufferMode

    # Test FIFO mode
    print("\n1. Testing FIFO mode...")
    buffer = StreamBuffer(max_size=5, mode=BufferMode.FIFO)

    for i in range(5):
        assert buffer.put(i), f"Failed to put {i}"
    assert buffer.is_full, "Buffer should be full"
    assert buffer.size == 5

    # Get items in order
    for i in range(5):
        item = buffer.get(timeout=0.1)
        assert item == i, f"Expected {i}, got {item}"
    assert buffer.is_empty, "Buffer should be empty"
    print("   FIFO mode: PASS")

    # Test DROP_OLD mode
    print("\n2. Testing DROP_OLD mode...")
    buffer = StreamBuffer(max_size=3, mode=BufferMode.DROP_OLD)
    for i in range(10):
        buffer.put(i)

    # Should have last 3 items
    items = []
    while not buffer.is_empty:
        items.append(buffer.get_nowait())
    assert items == [7, 8, 9], f"Expected [7, 8, 9], got {items}"
    assert buffer.items_dropped == 7, f"Expected 7 dropped, got {buffer.items_dropped}"
    print(f"   DROP_OLD mode: PASS (dropped {buffer.items_dropped} items)")

    # Test stop signal
    print("\n3. Testing stop signal...")
    buffer = StreamBuffer(max_size=10)
    buffer.put(1)
    buffer.stop()
    assert buffer.put(2) == False, "Should not accept after stop"
    print("   Stop signal: PASS")

    print("\nStreamBuffer: ALL TESTS PASSED")


def test_model_downloader():
    """Test ModelDownloader component."""
    print("\n" + "="*60)
    print("TEST: ModelDownloader")
    print("="*60)

    from models.downloader import ModelDownloader, SourceType

    with tempfile.TemporaryDirectory() as tmp:
        downloader = ModelDownloader(cache_dir=Path(tmp))

        # Test source type detection
        print("\n1. Testing source type detection...")
        assert downloader.detect_source_type("/path/to/model.pt") == SourceType.LOCAL
        assert downloader.detect_source_type("https://example.com/model.pt") == SourceType.URL
        assert downloader.detect_source_type("s3://bucket/model.pt") == SourceType.S3
        assert downloader.detect_source_type("ultralytics/yolov8n") == SourceType.HUGGINGFACE
        print("   Source detection: PASS")

        # Test local file handling
        print("\n2. Testing local file handling...")
        test_file = Path(tmp) / "test_model.pt"
        test_file.write_bytes(b"fake model data for testing")

        result = downloader.download(str(test_file), "test_local")
        assert result.success, f"Download failed: {result.error}"
        assert result.source_type == SourceType.LOCAL
        print(f"   Local file: PASS (path={result.path})")

        # Test cache checking
        print("\n3. Testing cache operations...")
        assert not downloader.is_cached("nonexistent")
        print("   Cache check: PASS")

        # Test checksum
        print("\n4. Testing checksum validation...")
        checksum = downloader.get_checksum(test_file)
        assert len(checksum) == 64, f"SHA256 should be 64 chars, got {len(checksum)}"
        assert downloader.validate_checksum(test_file, checksum)
        assert not downloader.validate_checksum(test_file, "invalid")
        print(f"   Checksum: PASS ({checksum[:16]}...)")

    print("\nModelDownloader: ALL TESTS PASSED")


def test_typed_results():
    """Test Typed Results component."""
    print("\n" + "="*60)
    print("TEST: Typed Results")
    print("="*60)

    from output.results import (
        BoundingBox, Detection, DetectionResults,
        Keypoint, KeypointSkeleton, PoseResults,
        create_detection_results
    )

    # Test BoundingBox
    print("\n1. Testing BoundingBox...")
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8)
    assert abs(bbox.width - 0.4) < 0.001
    assert abs(bbox.height - 0.6) < 0.001
    assert abs(bbox.area - 0.24) < 0.001

    # Test format conversions
    xywh = bbox.to_xywh()
    assert abs(xywh[2] - 0.4) < 0.001  # width

    cxcywh = bbox.to_cxcywh()
    assert abs(cxcywh[0] - 0.3) < 0.001  # center_x

    pixels = bbox.to_pixels(1920, 1080)
    assert pixels == (192, 216, 960, 864)
    print("   BoundingBox: PASS")

    # Test Detection
    print("\n2. Testing Detection...")
    det = Detection(
        class_id=0,
        class_name="PERSON",
        confidence=0.95,
        bbox=bbox
    )
    det_dict = det.to_dict()
    assert det_dict["class_name"] == "PERSON"
    assert det_dict["confidence"] == 0.95
    print("   Detection: PASS")

    # Test DetectionResults
    print("\n3. Testing DetectionResults...")
    results = DetectionResults(
        frame_number=100,
        timestamp_ms=3333,
        inference_time_ms=15.5,
        model_id="yolo_v8",
        detections=[
            det,
            Detection(class_id=1, class_name="BALL", confidence=0.8,
                     bbox=BoundingBox(0.4, 0.4, 0.5, 0.5)),
            Detection(class_id=0, class_name="PERSON", confidence=0.6,
                     bbox=BoundingBox(0.6, 0.2, 0.9, 0.9)),
        ]
    )

    assert results.count == 3
    assert results.count_by_class() == {"PERSON": 2, "BALL": 1}
    print(f"   DetectionResults: PASS (count={results.count})")

    # Test filtering
    print("\n4. Testing filtering...")
    filtered = results.filter(classes=["person"], min_confidence=0.7)
    assert filtered.count == 1
    assert filtered.detections[0].confidence == 0.95

    filtered2 = results.filter(max_detections=2)
    assert filtered2.count == 2
    print("   Filtering: PASS")

    # Test serialization
    print("\n5. Testing serialization...")
    json_str = results.to_json()
    parsed = json.loads(json_str)
    assert parsed["detection_count"] == 3
    assert "detections" in parsed

    # Test from_dict
    restored = DetectionResults.from_dict(parsed)
    assert restored.count == 3
    print("   Serialization: PASS")

    # Test YOLO format export
    print("\n6. Testing YOLO format export...")
    yolo_lines = results.to_yolo_format(1920, 1080)
    assert len(yolo_lines) == 3
    assert yolo_lines[0].startswith("0 ")  # class_id 0
    print(f"   YOLO format: PASS ({len(yolo_lines)} lines)")

    # Test factory function
    print("\n7. Testing factory function...")
    results2 = create_detection_results(
        frame_number=1,
        detections=[
            {"class_id": 0, "class_name": "PERSON", "confidence": 0.9,
             "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5}}
        ]
    )
    assert results2.count == 1
    print("   Factory function: PASS")

    # Test IoU calculation
    print("\n8. Testing IoU calculation...")
    bbox1 = BoundingBox(0.0, 0.0, 0.5, 0.5)
    bbox2 = BoundingBox(0.25, 0.25, 0.75, 0.75)
    iou = bbox1.iou(bbox2)
    assert 0.1 < iou < 0.2, f"IoU should be ~0.14, got {iou}"
    print(f"   IoU calculation: PASS (iou={iou:.3f})")

    print("\nTyped Results: ALL TESTS PASSED")


def test_buffered_frame_provider():
    """Test BufferedFrameProvider with synthetic frames."""
    print("\n" + "="*60)
    print("TEST: BufferedFrameProvider")
    print("="*60)

    from core.stream_buffer import BufferedFrameProvider, BufferMode

    # Simulate a frame source
    def frame_generator(n_frames=50, delay=0.01):
        for i in range(n_frames):
            time.sleep(delay)
            yield {"frame_number": i, "data": f"frame_{i}"}

    print("\n1. Testing buffered iteration...")
    provider = BufferedFrameProvider(
        frame_source=frame_generator(30),
        buffer_size=10,
        mode=BufferMode.FIFO
    )
    provider.start()

    frames_received = []
    for frame in provider.iterate(timeout=0.5):
        frames_received.append(frame)
        if len(frames_received) >= 30:
            break

    provider.stop()

    assert len(frames_received) == 30, f"Expected 30 frames, got {len(frames_received)}"
    # Check order preserved
    for i, frame in enumerate(frames_received):
        assert frame["frame_number"] == i, f"Frame order mismatch at {i}"

    print(f"   Buffered iteration: PASS ({len(frames_received)} frames)")
    print(f"   Stats: {provider.stats}")

    # Test DROP_OLD mode
    print("\n2. Testing DROP_OLD mode under backpressure...")

    def slow_consumer():
        """Simulate slow processing"""
        time.sleep(0.05)

    provider2 = BufferedFrameProvider(
        frame_source=frame_generator(50, delay=0.005),  # Fast producer
        buffer_size=5,
        mode=BufferMode.DROP_OLD
    )
    provider2.start()

    frames_received2 = []
    for frame in provider2.iterate(timeout=0.5):
        slow_consumer()  # Simulate slow processing
        frames_received2.append(frame)
        if len(frames_received2) >= 20:
            break

    provider2.stop()
    stats = provider2.stats
    print(f"   DROP_OLD stats: received={len(frames_received2)}, dropped={stats['items_dropped']}")
    print("   BufferedFrameProvider: PASS")

    print("\nBufferedFrameProvider: ALL TESTS PASSED")


def test_yolo_model_integration():
    """Test YOLOModel with new typed results (requires ultralytics)."""
    print("\n" + "="*60)
    print("TEST: YOLOModel Integration")
    print("="*60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("   SKIPPED: ultralytics not installed")
        return

    # Check for test model
    model_path = Path("/Users/spectatr/Downloads/Inference2.0/infer_2.0/temp/yolov8n.pt")
    if not model_path.exists():
        print(f"   SKIPPED: Test model not found at {model_path}")
        return

    from models.yolo_model import YOLOModel
    from output.results import DetectionResults

    print("\n1. Loading model...")
    model = YOLOModel(device="cpu", half_precision=False)
    success = model.load(str(model_path))
    assert success, "Failed to load model"
    print(f"   Model loaded: {len(model.class_names)} classes")

    print("\n2. Running inference on synthetic frame...")
    # Create a synthetic frame (random noise)
    frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    results = model.predict([frame], confidence=0.1)

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert isinstance(results[0], DetectionResults), "Should return DetectionResults"

    result = results[0]
    print(f"   Detections: {result.count}")
    print(f"   By class: {result.count_by_class()}")

    print("\n3. Testing filtering on results...")
    if result.count > 0:
        filtered = result.filter(min_confidence=0.5)
        print(f"   Filtered (conf>=0.5): {filtered.count} detections")

    print("\n4. Testing serialization...")
    json_output = result.to_json()
    parsed = json.loads(json_output)
    assert "detections" in parsed
    assert "detection_count" in parsed
    print("   Serialization: PASS")

    print("\nYOLOModel Integration: ALL TESTS PASSED")


def test_full_pipeline():
    """Test full pipeline: BufferedFrameProvider -> Model -> TypedResults."""
    print("\n" + "="*60)
    print("TEST: Full Pipeline")
    print("="*60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("   SKIPPED: ultralytics not installed")
        return

    model_path = Path("/Users/spectatr/Downloads/Inference2.0/infer_2.0/temp/yolov8n.pt")
    if not model_path.exists():
        print(f"   SKIPPED: Test model not found")
        return

    from core.stream_buffer import BufferedFrameProvider, BufferMode
    from models.yolo_model import YOLOModel
    from output.results import DetectionResults

    # Simulate frame source
    def synthetic_frames(n_frames=10):
        for i in range(n_frames):
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            yield {"frame_number": i, "frame": frame}

    print("\n1. Setting up pipeline...")
    model = YOLOModel(device="cpu")
    model.load(str(model_path))

    provider = BufferedFrameProvider(
        frame_source=synthetic_frames(10),
        buffer_size=5,
        mode=BufferMode.FIFO
    )
    provider.start()

    print("\n2. Processing frames through pipeline...")
    all_results = []
    for packet in provider.iterate(timeout=1.0):
        # Run inference
        results = model.predict([packet["frame"]], confidence=0.1)
        result = results[0]

        # Store result
        all_results.append({
            "frame_number": packet["frame_number"],
            "detection_count": result.count,
            "detections": result.to_dict()["detections"]
        })

        print(f"   Frame {packet['frame_number']}: {result.count} detections")

    provider.stop()

    print(f"\n3. Pipeline complete: processed {len(all_results)} frames")
    total_detections = sum(r["detection_count"] for r in all_results)
    print(f"   Total detections: {total_detections}")

    print("\nFull Pipeline: ALL TESTS PASSED")


def main():
    """Run all integration tests."""
    print("\n" + "#"*60)
    print("# INTEGRATION TESTS FOR NEW COMPONENTS")
    print("#"*60)

    tests = [
        ("StreamBuffer", test_stream_buffer),
        ("ModelDownloader", test_model_downloader),
        ("Typed Results", test_typed_results),
        ("BufferedFrameProvider", test_buffered_frame_provider),
        ("YOLOModel Integration", test_yolo_model_integration),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, "PASS", None))
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, status, error in results:
        status_str = f"\033[92mPASS\033[0m" if status == "PASS" else f"\033[91mFAIL\033[0m"
        print(f"  {name}: {status_str}")
        if error:
            print(f"    Error: {error}")

    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
