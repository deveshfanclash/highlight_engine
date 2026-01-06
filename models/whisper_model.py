"""
Whisper Model Wrapper

Wrapper for OpenAI Whisper audio transcription models.
Supports both local whisper and transformers implementations.
"""

import time
import logging
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

from models.base_model import BaseModel, ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Single transcription segment"""
    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 1.0
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 4),
            "language": self.language
        }


@dataclass
class TranscriptionOutput:
    """Output from Whisper transcription"""
    text: str  # Full text
    segments: List[TranscriptionSegment]
    language: str
    inference_time_ms: float = 0.0
    model_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "segments": [s.to_dict() for s in self.segments],
            "inference_time_ms": round(self.inference_time_ms, 1),
            "model_id": self.model_id
        }


class WhisperModel(BaseModel):
    """
    Wrapper for OpenAI Whisper models.

    Supports:
    - transformers library (preferred)
    - openai-whisper library

    Usage:
        model = WhisperModel(
            model_id="commentary_whisper",
            model_name="openai/whisper-base",
            device="cuda:0"
        )
        model.load()
        output = model.transcribe(audio_array, sample_rate=16000)
    """

    def __init__(
        self,
        model_id: str,
        model_name: str = "openai/whisper-base",
        device: str = "cpu",
        language: str = "en",
        task: str = "transcribe",
        class_mapping: Optional[Dict[int, str]] = None
    ):
        """
        Initialize Whisper model.

        Args:
            model_id: Unique identifier
            model_name: Model name/path (e.g., "openai/whisper-base")
            device: Device to run on
            language: Language code
            task: "transcribe" or "translate"
        """
        super().__init__(model_id, device, class_mapping)
        self.model_name = model_name
        self.language = language
        self.task = task
        self._processor = None
        self._use_transformers = True

    def load(self, model_path: str = None) -> bool:
        """
        Load Whisper model.

        Args:
            model_path: Optional local path (ignored if model_name is set)

        Returns:
            True if loaded successfully
        """
        # Try transformers first
        try:
            return self._load_transformers()
        except ImportError:
            pass

        # Fall back to openai-whisper
        try:
            return self._load_openai_whisper()
        except ImportError:
            raise ImportError(
                "Either transformers or openai-whisper is required. "
                "Install with: pip install transformers or pip install openai-whisper"
            )

    def _load_transformers(self) -> bool:
        """Load using transformers library"""
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch

        logger.info(f"Loading Whisper model: {self.model_name}")

        self._processor = WhisperProcessor.from_pretrained(self.model_name)
        self._model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

        # Move to device
        if "cuda" in self.device:
            device_id = self.device.split(":")[-1] if ":" in self.device else "0"
            self._model = self._model.to(f"cuda:{device_id}")
        elif self.device == "mps":
            self._model = self._model.to("mps")

        self._use_transformers = True
        self._loaded = True
        logger.info(f"Whisper model loaded (transformers)")
        return True

    def _load_openai_whisper(self) -> bool:
        """Load using openai-whisper library"""
        import whisper

        # Extract model size from name
        # "openai/whisper-base" -> "base"
        model_size = self.model_name.split("-")[-1] if "-" in self.model_name else "base"

        logger.info(f"Loading Whisper model: {model_size}")
        self._model = whisper.load_model(model_size, device=self.device)

        self._use_transformers = False
        self._loaded = True
        logger.info(f"Whisper model loaded (openai-whisper)")
        return True

    def predict(
        self,
        frames: List[np.ndarray],
        confidence_threshold: float = 0.5,
        classes: Optional[List[int]] = None
    ) -> List[ModelOutput]:
        """
        BaseModel interface - not used for audio models.
        Use transcribe() instead.
        """
        raise NotImplementedError(
            "Whisper is an audio model. Use transcribe() instead of predict()"
        )

    def get_class_names(self) -> Dict[int, str]:
        """Audio models don't have classes"""
        return {}

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        sample_rate: int = 16000,
        return_timestamps: bool = True
    ) -> TranscriptionOutput:
        """
        Transcribe audio.

        Args:
            audio: Audio array (mono, float32) or path to audio file
            sample_rate: Sample rate of audio (default 16000 Hz)
            return_timestamps: Include word/segment timestamps

        Returns:
            TranscriptionOutput with text and segments
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        if self._use_transformers:
            result = self._transcribe_transformers(audio, sample_rate, return_timestamps)
        else:
            result = self._transcribe_openai(audio, sample_rate, return_timestamps)

        inference_time = (time.time() - start_time) * 1000
        result.inference_time_ms = inference_time
        result.model_id = self.model_id

        return result

    def _transcribe_transformers(
        self,
        audio: Union[np.ndarray, str],
        sample_rate: int,
        return_timestamps: bool
    ) -> TranscriptionOutput:
        """Transcribe using transformers"""
        import torch

        # Load audio if path provided
        if isinstance(audio, str):
            import librosa
            audio, sample_rate = librosa.load(audio, sr=16000)

        # Process audio
        inputs = self._processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        # Move to device
        if "cuda" in self.device:
            inputs = inputs.to(self._model.device)

        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(
                inputs["input_features"],
                language=self.language,
                task=self.task,
                return_timestamps=return_timestamps
            )

        # Decode
        transcription = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # For now, return single segment (timestamps require more complex handling)
        segments = [TranscriptionSegment(
            text=transcription,
            start=0.0,
            end=len(audio) / sample_rate
        )]

        return TranscriptionOutput(
            text=transcription,
            segments=segments,
            language=self.language
        )

    def _transcribe_openai(
        self,
        audio: Union[np.ndarray, str],
        sample_rate: int,
        return_timestamps: bool
    ) -> TranscriptionOutput:
        """Transcribe using openai-whisper"""

        # Load audio if array provided
        if isinstance(audio, np.ndarray):
            # openai-whisper expects float32 audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

        result = self._model.transcribe(
            audio,
            language=self.language,
            task=self.task,
            verbose=False
        )

        # Parse segments
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptionSegment(
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
                confidence=seg.get("avg_logprob", 1.0)
            ))

        return TranscriptionOutput(
            text=result["text"],
            segments=segments,
            language=result.get("language", self.language)
        )

    def transcribe_chunks(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int = 16000,
        chunk_duration_sec: float = 30.0
    ) -> List[TranscriptionOutput]:
        """
        Transcribe multiple audio chunks.

        Args:
            audio_chunks: List of audio arrays
            sample_rate: Sample rate
            chunk_duration_sec: Duration of each chunk in seconds

        Returns:
            List of TranscriptionOutput, one per chunk
        """
        results = []
        for i, chunk in enumerate(audio_chunks):
            result = self.transcribe(chunk, sample_rate)
            # Adjust timestamps for chunk position
            chunk_offset = i * chunk_duration_sec
            for seg in result.segments:
                seg.start += chunk_offset
                seg.end += chunk_offset
            results.append(result)
        return results


# =============================================================================
# FACTORY
# =============================================================================

def create_whisper_model(
    model_id: str,
    model_name: str = "openai/whisper-base",
    device: str = "cpu",
    language: str = "en",
    task: str = "transcribe"
) -> WhisperModel:
    """
    Factory function to create and load a Whisper model.

    Args:
        model_id: Unique identifier
        model_name: Model name (e.g., "openai/whisper-base")
        device: Device to run on
        language: Language code
        task: "transcribe" or "translate"

    Returns:
        Loaded WhisperModel instance
    """
    model = WhisperModel(
        model_id=model_id,
        model_name=model_name,
        device=device,
        language=language,
        task=task
    )
    model.load()
    return model
