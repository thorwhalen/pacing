"""
Transcription interfaces for the PACING platform.

These interfaces define how audio is converted to text. Implementations can use
various speech-to-text services (Deepgram, Whisper, Google Speech, etc.) or
mock transcribers for testing.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import numpy as np

from pacing.models.data_models import TranscriptionResult


class ITranscriber(ABC):
    """
    Abstract interface for speech-to-text transcription.

    This interface allows the system to support multiple transcription backends
    without changing the core logic. The transcriber is responsible for:

    1. Converting audio chunks to text
    2. Providing confidence scores for transcriptions
    3. Handling partial (streaming) transcriptions
    4. Speaker diarization (if supported)

    Design Philosophy:
    - Transcribers should be stateless or manage their own state
    - They should handle their own buffering and context management
    - Confidence scores must be normalized to [0.0, 1.0]
    """

    @abstractmethod
    async def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        is_final: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe a single audio chunk.

        Args:
            audio_chunk: Audio samples (typically float32 or int16)
            sample_rate: Sample rate in Hz
            is_final: Whether this is the final chunk in a sequence

        Returns:
            TranscriptionResult: The transcription with confidence score

        Notes:
            - For streaming transcription, is_final=False produces partial results
            - Implementations should handle silence gracefully
            - Empty audio should return empty text with high confidence
        """
        pass

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        sample_rate: int
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Transcribe a stream of audio chunks.

        This is a convenience method that processes an audio stream and yields
        transcription results. The default implementation calls transcribe_chunk()
        for each audio chunk.

        Args:
            audio_stream: Async iterator of audio chunks
            sample_rate: Sample rate in Hz

        Yields:
            TranscriptionResult: Transcriptions as they become available

        Example:
            async for result in transcriber.transcribe_stream(audio_stream, 16000):
                print(f"{result.text} (confidence: {result.confidence_score})")
        """
        async for chunk in audio_stream:
            result = await self.transcribe_chunk(chunk, sample_rate, is_final=False)
            if result.text:  # Only yield non-empty transcriptions
                yield result

    @abstractmethod
    def supports_speaker_diarization(self) -> bool:
        """
        Check if this transcriber supports speaker diarization.

        Returns:
            bool: True if speaker_id will be populated in TranscriptionResult
        """
        pass

    def get_model_info(self) -> dict:
        """
        Get information about the transcription model.

        Returns:
            dict: Model metadata (name, version, language, etc.)
        """
        return {
            "name": self.__class__.__name__,
            "version": "unknown",
            "language": "en-US"
        }
