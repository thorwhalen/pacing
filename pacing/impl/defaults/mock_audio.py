"""
Mock audio provider for testing and demonstration.

This implementation generates synthetic audio or plays back pre-scripted scenarios
without requiring actual audio hardware.
"""

import asyncio
import time
from typing import Iterator, AsyncIterator, List
import numpy as np

from pacing.core.audio_interfaces import IAudioProvider


class MockAudioProvider(IAudioProvider):
    """
    Mock audio provider that generates synthetic audio signals.

    This implementation is useful for:
    - Testing the platform without audio hardware
    - Demonstrating the system in controlled scenarios
    - Development and CI/CD pipelines

    The mock provider can:
    - Generate silence
    - Generate synthetic tones
    - Simulate realistic audio chunk timing
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        total_duration_sec: float = 60.0
    ):
        """
        Initialize the mock audio provider.

        Args:
            sample_rate: Sample rate in Hz (default: 16000)
            chunk_duration_ms: Duration of each chunk in milliseconds
            total_duration_sec: Total duration to stream before stopping
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.total_duration_sec = total_duration_sec
        self._is_streaming = False
        self._chunk_size = int(sample_rate * chunk_duration_ms / 1000)

    def start_stream(self) -> None:
        """Start the mock audio stream."""
        self._is_streaming = True
        print(f"[MockAudioProvider] Started streaming at {self.sample_rate}Hz")

    def stop_stream(self) -> None:
        """Stop the mock audio stream."""
        self._is_streaming = False
        print("[MockAudioProvider] Stopped streaming")

    def get_audio_chunks(self) -> Iterator[np.ndarray]:
        """
        Generate mock audio chunks.

        Yields:
            np.ndarray: Synthetic audio data (float32, normalized to [-1, 1])
        """
        if not self._is_streaming:
            raise RuntimeError("Stream not started. Call start_stream() first.")

        start_time = time.time()
        chunk_interval = self.chunk_duration_ms / 1000.0

        while self._is_streaming:
            elapsed = time.time() - start_time
            if elapsed >= self.total_duration_sec:
                break

            # Generate a chunk of silence (could be enhanced with synthetic speech)
            chunk = np.zeros(self._chunk_size, dtype=np.float32)

            # Add very low amplitude noise to simulate realistic audio
            chunk += np.random.normal(0, 0.001, self._chunk_size).astype(np.float32)

            yield chunk

            # Simulate real-time audio capture timing
            time.sleep(chunk_interval)

    async def get_audio_chunks_async(self) -> AsyncIterator[np.ndarray]:
        """
        Asynchronously generate mock audio chunks.

        Yields:
            np.ndarray: Synthetic audio data
        """
        if not self._is_streaming:
            raise RuntimeError("Stream not started. Call start_stream() first.")

        start_time = asyncio.get_event_loop().time()
        chunk_interval = self.chunk_duration_ms / 1000.0

        while self._is_streaming:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= self.total_duration_sec:
                break

            # Generate chunk
            chunk = np.zeros(self._chunk_size, dtype=np.float32)
            chunk += np.random.normal(0, 0.001, self._chunk_size).astype(np.float32)

            yield chunk

            # Simulate real-time timing asynchronously
            await asyncio.sleep(chunk_interval)

    def get_sample_rate(self) -> int:
        """Get the sample rate."""
        return self.sample_rate

    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming


class ScriptedAudioProvider(IAudioProvider):
    """
    Audio provider that plays back pre-scripted scenarios.

    This is useful for creating reproducible demonstrations where specific
    "conversations" need to occur at specific times.
    """

    def __init__(
        self,
        script: List[tuple[float, str]],
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100
    ):
        """
        Initialize with a script.

        Args:
            script: List of (timestamp, audio_description) tuples
                   e.g., [(0.0, "silence"), (5.0, "tone_440hz"), (10.0, "silence")]
            sample_rate: Sample rate in Hz
            chunk_duration_ms: Duration of each chunk in milliseconds
        """
        self.script = script
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self._is_streaming = False
        self._chunk_size = int(sample_rate * chunk_duration_ms / 1000)

    def start_stream(self) -> None:
        """Start playback."""
        self._is_streaming = True

    def stop_stream(self) -> None:
        """Stop playback."""
        self._is_streaming = False

    def get_audio_chunks(self) -> Iterator[np.ndarray]:
        """Generate audio according to script."""
        if not self._is_streaming:
            raise RuntimeError("Stream not started.")

        for timestamp, description in self.script:
            if not self._is_streaming:
                break

            # Wait until timestamp
            time.sleep(timestamp)

            # Generate appropriate audio based on description
            if description == "silence":
                chunk = np.zeros(self._chunk_size, dtype=np.float32)
            else:
                # Placeholder for actual audio generation
                chunk = np.random.normal(0, 0.01, self._chunk_size).astype(np.float32)

            yield chunk

    def get_sample_rate(self) -> int:
        """Get the sample rate."""
        return self.sample_rate

    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming
