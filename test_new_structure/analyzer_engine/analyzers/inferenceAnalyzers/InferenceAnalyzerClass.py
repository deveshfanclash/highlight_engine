from analyzer_engine.analyzers.BaseAnalyzerClass import BaseAnalyzer

class InferenceAnalyzer(BaseAnalyzer):
    """
    Base class for inference analyzers.
    Extend this class and implement predict for custom inference logic.
    """

    def __init__(self, streamUrl, matchId, **params):
        super().__init__(streamUrl, matchId, **params)

    def process_stream(self):
        """
        Main stream processing method.
        This method orchestrates the inference prediction, 
        but must be extended for analyzer-specific workflows.
        """
        # Example flow: call predict and (optionally) post-process/persist results.
        prediction = self.predict()
        return prediction

    def predict(self):
        """
        Abstract method for inference/prediction logic.
        Subclasses must override this.
        """
        raise NotImplementedError("Subclasses must implement the predict method.")

