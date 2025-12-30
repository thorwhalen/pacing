"""
Session stream interfaces for the PACING platform.

The session stream is the central orchestrator that connects audio providers,
transcribers, and sidecar agents.
"""

from abc import ABC, abstractmethod
from typing import List, AsyncIterator, Optional

from pacing.models.data_models import TranscriptionResult, SessionMetadata
from pacing.core.agent_interfaces import ISidecarAgent
from pacing.core.audio_interfaces import IAudioProvider
from pacing.core.transcription_interfaces import ITranscriber


class ISessionStream(ABC):
    """
    Abstract interface for the live session stream orchestrator.

    The SessionStream is responsible for:
    1. Coordinating audio capture, transcription, and agent processing
    2. Managing the session lifecycle (start, stop, pause)
    3. Broadcasting transcription updates to all registered agents
    4. Handling errors and recovery

    Design Philosophy:
    - The stream is the "nervous system" of the live mode
    - It enforces privacy boundaries (agents don't see raw audio)
    - It's observable (agents subscribe to it)
    - It's composable (add/remove agents dynamically)
    """

    @abstractmethod
    def register_agent(self, agent: ISidecarAgent) -> None:
        """
        Register a sidecar agent to receive transcription updates.

        Args:
            agent: The agent to register

        Notes:
            - Agents can be registered before or during a session
            - Multiple agents of the same type can be registered
        """
        pass

    @abstractmethod
    def unregister_agent(self, agent: ISidecarAgent) -> None:
        """
        Unregister a sidecar agent.

        Args:
            agent: The agent to unregister
        """
        pass

    @abstractmethod
    async def start_session(
        self,
        session_metadata: SessionMetadata,
        audio_provider: IAudioProvider,
        transcriber: ITranscriber
    ) -> None:
        """
        Start a new clinical session.

        Args:
            session_metadata: Session information
            audio_provider: Source of audio data
            transcriber: Speech-to-text engine

        Raises:
            SessionError: If session cannot be started
        """
        pass

    @abstractmethod
    async def stop_session(self) -> None:
        """
        Stop the current session and finalize processing.

        This method:
        1. Stops audio capture
        2. Flushes any pending transcriptions
        3. Notifies all agents that the session has ended
        4. Cleans up resources
        """
        pass

    @abstractmethod
    def get_transcription_stream(self) -> AsyncIterator[TranscriptionResult]:
        """
        Get an async iterator of transcription results.

        This allows external observers to monitor transcriptions without
        registering as full sidecar agents.

        Yields:
            TranscriptionResult: Transcriptions as they occur
        """
        pass

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """
        Check if a session is currently active.

        Returns:
            bool: True if session is running
        """
        pass

    @property
    def registered_agents(self) -> List[ISidecarAgent]:
        """
        Get list of currently registered agents.

        Returns:
            List[ISidecarAgent]: All registered agents
        """
        return []
