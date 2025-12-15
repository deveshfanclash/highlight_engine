from analyzer_engine.analyzers.BaseAnalyzerClass import BaseAnalyzer
from analyzer_engine.services.DynamoDbService import DynamoDbService
from analyzer_engine.schemas.analyzer_schema import ODConfigSchema, ODSchema
import requests
import os

class ODAnalyzer(BaseAnalyzer):
    def __init__(self, streamUrl, matchId, **params):
        super().__init__(streamUrl, matchId, **params)
        # Interpret params using ODConfigSchema
        self.od_config = ODConfigSchema(**self.params)
        self.model_path = self.od_config.model_path
        self.local_model_path = self._download_model(self.model_path)
        self.dynamo_service = DynamoDbService()

    def _download_model(self, model_url):
        # Download the model file from the URL specified in the params
        local_filename = model_url.split('/')[-1]
        local_path = os.path.join("/tmp", local_filename)
        if not os.path.exists(local_path):
            # Stream download
            r = requests.get(model_url, stream=True)
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    # INSERT_YOUR_CODE
from analyzer_engine.analyzers.inferenceAnalyzers.InferenceAnalyzerClass import InferenceAnalyzer

class ODAnalyzer(InferenceAnalyzer):
    def __init__(self, streamUrl, matchId, **params):
        super().__init__(streamUrl, matchId, **params)
        # Interpret params using ODConfigSchema
        self.od_config = ODConfigSchema(**self.params)
        self.model_path = self.od_config.model_path
        self.local_model_path = self._download_model(self.model_path)
        self.dynamo_service = DynamoDbService()

    def _download_model(self, model_url):
        # Download the model file from the URL specified in the params
        local_filename = model_url.split('/')[-1]
        local_path = os.path.join("/tmp", local_filename)
        if not os.path.exists(local_path):
            # Stream download
            r = requests.get(model_url, stream=True)
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path

    def predict(self):
        """
        Prediction logic for ODAnalyzer.
        This should run the OD inference and return the detections.
        """
        # This is a mock implementation. Replace with real OD inference logic.
        detections = [
            ODSchema(
                match_id=self.matchId,
                object_label="car",
                confidence=0.97,
                bbox=[100, 200, 150, 250]
            )
        ]
        return detections

    def process_stream(self):
        """
        This method processes the stream indicated by self.streamUrl.
        Detected objects are dumped in DynamoDB using ODSchema format.
        """
        # INSERT_YOUR_CODE
        # Read the video stream
        self.read_stream()  # Provided by BaseAnalyzer, sets self.cv

        # Run prediction (process a frame or the stream)
        detections = self.predict()

        # This is a mock implementation. Actual OD logic goes here.
        # Assume 'detections' is a list of detection results conforming to ODSchema.
        detections = [
            ODSchema(
                match_id=self.matchId,
                object_label="car",
                confidence=0.97,
                bbox=[100, 200, 150, 250]
            )
        ]
        for detection in detections:
            # Serialize result with ODSchema, then save to DynamoDB.
            item = detection.dict()
            self.dynamo_service.save_item(item)
