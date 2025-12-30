"""
Mock transcriber for testing and demonstration.

This implementation returns pre-scripted text with simulated latency and
confidence scores, without requiring actual speech-to-text services.
"""

import asyncio
import random
from datetime import datetime
from typing import List, Optional
import numpy as np

from pacing.core.transcription_interfaces import ITranscriber
from pacing.models.data_models import TranscriptionResult


class MockTranscriber(ITranscriber):
    """
    Mock transcriber that yields pre-scripted text.

    This implementation simulates a real transcription service by:
    - Returning text from a pre-defined script
    - Adding realistic latency (processing delay)
    - Generating plausible confidence scores
    - Supporting partial (streaming) results

    Useful for:
    - Testing the platform without API costs
    - Creating reproducible demonstrations
    - Development and CI/CD
    """

    def __init__(
        self,
        script: Optional[List[str]] = None,
        latency_ms: float = 50.0,
        base_confidence: float = 0.85,
        confidence_variance: float = 0.10
    ):
        """
        Initialize the mock transcriber.

        Args:
            script: List of text strings to return in sequence
                   If None, uses a default demonstration script
            latency_ms: Simulated processing latency in milliseconds
            base_confidence: Base confidence score (0.0-1.0)
            confidence_variance: Random variance in confidence (+/-)
        """
        self.script = script or self._default_script()
        self.latency_ms = latency_ms
        self.base_confidence = base_confidence
        self.confidence_variance = confidence_variance
        self._script_index = 0

    def _default_script(self) -> List[str]:
        """
        Default demonstration script for a counseling session.

        Returns:
            List[str]: Pre-scripted conversation segments
        """
        return [
            "Hi, how have you been doing this week?",
            "I've been okay, but I had a really tough day on Tuesday.",
            "What happened on Tuesday?",
            "I lost my job. They said it was due to budget cuts.",
            "I'm sorry to hear that. That must be very stressful.",
            "Yeah, it's been hard. I almost relapsed that night.",
            "But you didn't?",
            "No, I called my sponsor instead.",
            "That's really good. You used your support system.",
            "I'm still taking my buprenorphine every day.",
            "How long have you been on that now?",
            "About six months.",
            "And it's been helping?",
            "Yeah, definitely. I haven't had any cravings in weeks.",
            "That's excellent progress.",
        ]

    async def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        is_final: bool = False
    ) -> TranscriptionResult:
        """
        Simulate transcription of an audio chunk.

        Args:
            audio_chunk: Audio samples (ignored in mock)
            sample_rate: Sample rate (ignored in mock)
            is_final: Whether this is the final chunk

        Returns:
            TranscriptionResult: Mock transcription with confidence score
        """
        # Simulate processing latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        # Get next text from script
        if self._script_index < len(self.script):
            text = self.script[self._script_index]
            self._script_index += 1
        else:
            # Script exhausted, return empty
            text = ""

        # Generate confidence score with variance
        confidence = self.base_confidence + random.uniform(
            -self.confidence_variance,
            self.confidence_variance
        )
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        # Occasionally generate low confidence to trigger auditor
        if random.random() < 0.15:  # 15% chance
            confidence = random.uniform(0.50, 0.70)

        return TranscriptionResult(
            text=text,
            timestamp=datetime.now(),
            confidence_score=confidence,
            is_partial=not is_final
        )

    def supports_speaker_diarization(self) -> bool:
        """Mock transcriber does not support speaker diarization."""
        return False

    def get_model_info(self) -> dict:
        """Get mock model information."""
        return {
            "name": "MockTranscriber",
            "version": "1.0.0",
            "language": "en-US",
            "type": "scripted"
        }

    def reset_script(self) -> None:
        """Reset the script index to start from the beginning."""
        self._script_index = 0


class AdaptiveConfidenceTranscriber(MockTranscriber):
    """
    Mock transcriber that adjusts confidence based on text characteristics.

    This variant simulates more realistic confidence scores by:
    - Lowering confidence for longer utterances (harder to transcribe)
    - Lowering confidence for utterances with medical/technical terms
    - Lowering confidence for utterances with negations (tricky)
    """

    DIFFICULT_TERMS = {
        "buprenorphine", "naloxone", "methadone", "suboxone",
        "opioid", "benzodiazepine", "relapse", "withdrawal"
    }

    async def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        is_final: bool = False
    ) -> TranscriptionResult:
        """Transcribe with adaptive confidence."""
        # Get base result
        result = await super().transcribe_chunk(audio_chunk, sample_rate, is_final)

        if not result.text:
            return result

        # Adjust confidence based on text characteristics
        confidence = result.confidence_score

        # Penalty for length
        word_count = len(result.text.split())
        if word_count > 15:
            confidence *= 0.90

        # Penalty for difficult terms
        text_lower = result.text.lower()
        if any(term in text_lower for term in self.DIFFICULT_TERMS):
            confidence *= 0.85

        # Penalty for negations (often misheard)
        if any(neg in text_lower for neg in ["not", "no", "never", "didn't"]):
            confidence *= 0.90

        confidence = max(0.40, min(1.0, confidence))

        # Create new result with adjusted confidence
        return TranscriptionResult(
            text=result.text,
            timestamp=result.timestamp,
            confidence_score=confidence,
            speaker_id=result.speaker_id,
            is_partial=result.is_partial
        )
