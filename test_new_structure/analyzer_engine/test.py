
from orchestrator import initialize_and_run_analyzers


if __name__ == "__main__":
    # Example requests for CameraViewChangeAnalyzer and ODAnalyzer
    test_requests = [
        {
            'analyzer_type': 'CameraViewChangeAnalyzer',
            'streamUrl': 'http://example.com/stream1',
            'streamType': 'live/Vod',
            'streamFormat': 'mp4/hls',
            'matchId': 'match_123',
            'sensitivity': 0.8  # Example of additional parameter
        },
        {
            'analyzer_type': 'ODAnalyzer',
            'streamUrl': 'http://example.com/stream2',
            'matchId': 456,
            'threshold': 0.5  # Another analyzer-specific parameter
        }
    ]

    try:
        results = initialize_and_run_analyzers(test_requests)
        for i, res in enumerate(results):
            print(f"Result from analyzer {test_requests[i]['analyzer_type']}: {res}")
    except Exception as e:
        print(f"Error occurred during orchestration: {e}")

