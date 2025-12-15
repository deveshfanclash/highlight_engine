from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    def __init__(self, streamUrl, matchId, **params):
        self.streamUrl = streamUrl
        self.matchId = matchId
        self.params = params
        self.validate_parameters()

    def validate_parameters(self):
        if not isinstance(self.streamUrl, str) or not self.streamUrl:
            raise ValueError("streamUrl must be a non-empty string")
        if not isinstance(self.matchId, (str, int)) or not self.matchId:
            raise ValueError("matchId must be a non-empty string or integer")
        # Additional parameters can be validated by subclasses if needed

    @abstractmethod
    def process_stream(self):
        """
        Abstract method to process the stream and save the results to the DB.
        Should be implemented by subclasses.
        """
        pass

    def read_stream(self):
    # INSERT_YOUR_CODE
        """
        Reads the video stream based on parameters such as streamType, streamFormat, etc.
        Sets self.cv (OpenCV VideoCapture object) for use in process_stream or other methods.
        """
        stream_type = self.params.get("streamType", "live").lower() if "streamType" in self.params else "live"
        stream_format = self.params.get("streamFormat", None)
        stream_url = self.streamUrl

        # Choose how to open the stream based on streamType/streamFormat
        # This is the canonical approach for handling rtsp/http/file etc.
        # You can expand this logic depending on your supported protocols/types.
        if stream_type in ["live", "rtsp", "hls"]:
            # For live sources (e.g., RTSP, HLS, direct camera stream)
            self.cv = self._open_video_stream(stream_url)
        elif stream_type in ["vod", "file"]:
            # For VOD or file input, treat streamUrl as a file path or downloadable URL
            self.cv = self._open_video_stream(stream_url)
        else:
            # Default fallback
            self.cv = self._open_video_stream(stream_url)

    def _open_video_stream(self, url):
        """
        Helper to open a video stream using OpenCV VideoCapture.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV (cv2) must be installed to use video stream reading functionality.")
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video stream: {url}")
        return cap
