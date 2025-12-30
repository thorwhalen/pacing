"""
Audio provider interfaces for the PACING platform.

These interfaces define how audio is captured and streamed into the system.
Implementations can range from microphone input to file playback to network streams.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator
import numpy as np


class IAudioProvider(ABC):
    """
    Abstract interface for audio input sources.

    This interface allows the system to be agnostic to the audio source,
    enabling different implementations for:
    - Live microphone input
    - Pre-recorded audio files
    - Network audio streams
    - Mock/synthetic audio for testing

    The audio provider is responsible for:
    1. Starting and stopping audio capture
    2. Yielding audio chunks at a consistent sample rate
    3. Handling audio device configuration
    """

    @abstractmethod
    def start_stream(self) -> None:
        """
        Initialize and start the audio stream.

        Raises:
            AudioDeviceError: If the audio device cannot be opened
            ConfigurationError: If audio parameters are invalid
        """
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """
        Stop the audio stream and release resources.

        This method should be idempotent (safe to call multiple times).
        """
        pass

    @abstractmethod
    def get_audio_chunks(self) -> Iterator[np.ndarray]:
        """
        Yield audio data chunks as they become available.

        Yields:
            np.ndarray: Audio samples (typically float32 or int16)

        Notes:
            - The sample rate should be consistent and queryable via get_sample_rate()
            - Chunk sizes may vary depending on the implementation
            - This is a blocking iterator; use get_audio_chunks_async() for async
        """
        pass

    async def get_audio_chunks_async(self) -> AsyncIterator[np.ndarray]:
        """
        Asynchronously yield audio data chunks.

        This is the preferred method for real-time streaming applications.

        Yields:
            np.ndarray: Audio samples

        Example:
            async for chunk in audio_provider.get_audio_chunks_async():
                await transcriber.transcribe_chunk(chunk)
        """
        # Default implementation wraps synchronous version
        # Subclasses should override for true async behavior
        for chunk in self.get_audio_chunks():
            yield chunk

    @abstractmethod
    def get_sample_rate(self) -> int:
        """
        Get the audio sample rate in Hz.

        Returns:
            int: Sample rate (e.g., 16000, 44100)
        """
        pass

    @property
    def is_streaming(self) -> bool:
        """
        Check if audio is currently being streamed.

        Returns:
            bool: True if stream is active
        """
        return False
