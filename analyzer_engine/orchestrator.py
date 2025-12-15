
from analyzer_engine.analyzers.ODAnalyzer.ODAnalyzerClass import ODAnalyzer

# You can extend this mapping as additional analyzers are implemented/imported
ANALYZER_REGISTRY = {
    "ODAnalyzer": ODAnalyzer,
    "CameraViewChangeAnalyzer": CameraViewChangeAnalyzer,
    # Example: "OtherAnalyzer": OtherAnalyzerClass,
    
}

def initialize_and_run_analyzers(analyzer_requests):
    """
    analyzer_requests: list of dict, each containing keys:
        - 'analyzer_type': str
        - 'streamUrl': str
        - 'matchId': str or int
        - ...additional analyzer-specific parameters
    """
    results = []
    for request in analyzer_requests:
        analyzer_type = request.get('analyzer_type')
        stream_url = request.get('streamUrl')
        match_id = request.get('matchId')
        params = {k: v for k, v in request.items() if k not in ['analyzer_type', 'streamUrl', 'matchId']}
        AnalyzerClass = ANALYZER_REGISTRY.get(analyzer_type)
        if AnalyzerClass is None:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")
        analyzer_instance = AnalyzerClass(stream_url, match_id, **params)
        res = analyzer_instance.process_stream()
        results.append(res)
    return results

    # INSERT_YOUR_CODE

