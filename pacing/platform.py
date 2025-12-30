"""
Main PACING platform orchestrator.

This module provides the primary entry point for initializing and configuring
the PACING platform with dependency injection.
"""

import asyncio
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime

from pacing.core.audio_interfaces import IAudioProvider
from pacing.core.transcription_interfaces import ITranscriber
from pacing.core.agent_interfaces import ISidecarAgent
from pacing.core.model_interfaces import IRiskModel, ISimulationModel
from pacing.core.session_interfaces import ISessionStream
from pacing.models.data_models import SessionMetadata, PatientGraph, TranscriptionResult


class OperatingMode(str, Enum):
    """
    Operating mode for the platform.

    DEV_MODE: Persists raw data (audio, transcripts) for debugging and auditing
    PROD_MODE: Ephemeral raw data, only structured extractions are retained
    """
    DEV_MODE = "dev"
    PROD_MODE = "prod"


class BasicSessionStream(ISessionStream):
    """
    Basic implementation of the session stream orchestrator.

    This implementation coordinates audio capture, transcription, and agent
    processing for a live clinical session.
    """

    def __init__(self, operating_mode: OperatingMode = OperatingMode.PROD_MODE):
        """
        Initialize the session stream.

        Args:
            operating_mode: DEV_MODE or PROD_MODE (affects data retention)
        """
        self.operating_mode = operating_mode
        self._agents: List[ISidecarAgent] = []
        self._is_active = False
        self._current_session: Optional[SessionMetadata] = None
        self._audio_provider: Optional[IAudioProvider] = None
        self._transcriber: Optional[ITranscriber] = None
        self._transcription_buffer: List[TranscriptionResult] = []

    def register_agent(self, agent: ISidecarAgent) -> None:
        """Register a sidecar agent."""
        if agent not in self._agents:
            self._agents.append(agent)
            print(f"[SessionStream] Registered agent: {agent.get_agent_name()}")

    def unregister_agent(self, agent: ISidecarAgent) -> None:
        """Unregister a sidecar agent."""
        if agent in self._agents:
            self._agents.remove(agent)
            print(f"[SessionStream] Unregistered agent: {agent.get_agent_name()}")

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
        """
        if self._is_active:
            raise RuntimeError("A session is already active. Stop it before starting a new one.")

        self._current_session = session_metadata
        self._audio_provider = audio_provider
        self._transcriber = transcriber
        self._is_active = True
        self._transcription_buffer = []

        # Notify all agents
        for agent in self._agents:
            agent.on_session_start(
                session_metadata.session_id,
                session_metadata.model_dump()
            )

        print(f"[SessionStream] Session started: {session_metadata.session_id}")

        # Start audio capture
        self._audio_provider.start_stream()

        # Begin processing audio asynchronously
        await self._process_audio_stream()

    async def _process_audio_stream(self) -> None:
        """Process audio stream and distribute transcriptions to agents."""
        if not self._audio_provider or not self._transcriber:
            return

        sample_rate = self._audio_provider.get_sample_rate()

        async for audio_chunk in self._audio_provider.get_audio_chunks_async():
            if not self._is_active:
                break

            # Transcribe the audio chunk
            transcription = await self._transcriber.transcribe_chunk(
                audio_chunk,
                sample_rate
            )

            # Store in buffer (if DEV_MODE, otherwise ephemeral)
            if self.operating_mode == OperatingMode.DEV_MODE:
                self._transcription_buffer.append(transcription)

            # Distribute to all agents
            await self._broadcast_transcription(transcription)

    async def _broadcast_transcription(self, transcription: TranscriptionResult) -> None:
        """Broadcast a transcription to all registered agents."""
        context = None
        if self._current_session:
            context = {
                "session_id": self._current_session.session_id,
                "patient_id": self._current_session.patient_id,
                "clinician_id": self._current_session.clinician_id
            }

        # Process agents in parallel
        tasks = [
            agent.on_transcription_update(transcription, context)
            for agent in self._agents
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_session(self) -> None:
        """Stop the current session."""
        if not self._is_active:
            return

        self._is_active = False

        # Stop audio capture
        if self._audio_provider:
            self._audio_provider.stop_stream()

        # Notify all agents
        if self._current_session:
            for agent in self._agents:
                agent.on_session_end(self._current_session.session_id)

            print(f"[SessionStream] Session ended: {self._current_session.session_id}")

        # Clear ephemeral data in PROD mode
        if self.operating_mode == OperatingMode.PROD_MODE:
            self._transcription_buffer = []
            print("[SessionStream] Ephemeral data cleared (PROD_MODE)")

        self._current_session = None
        self._audio_provider = None
        self._transcriber = None

    async def get_transcription_stream(self):
        """Get transcription stream (async generator)."""
        # This is a simplified implementation
        # In production, use asyncio.Queue for better flow control
        for transcription in self._transcription_buffer:
            yield transcription

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self._is_active

    @property
    def registered_agents(self) -> List[ISidecarAgent]:
        """Get registered agents."""
        return self._agents.copy()


class PacingPlatform:
    """
    Main PACING platform class.

    This is the primary interface for initializing and configuring the PACING
    system. It provides dependency injection for all major components:
    - Audio providers
    - Transcribers
    - Sidecar agents
    - Risk models
    - Operating mode (DEV/PROD)

    Example Usage:
        # Initialize platform in DEV mode
        platform = PacingPlatform(operating_mode=OperatingMode.DEV_MODE)

        # Inject custom components
        platform.set_transcriber(DeepgramTranscriber(api_key=...))
        platform.set_risk_model(MyBayesianNetwork())

        # Register agents
        platform.register_agent(UncertaintyAuditor(confidence_threshold=0.70))
        platform.register_agent(GuideAgent())
        platform.register_agent(ScribeAgent())

        # Start a session
        session = SessionMetadata(
            session_id="session_001",
            patient_id="patient_123",
            clinician_id="clinician_456",
            start_time=datetime.now()
        )

        audio = MyAudioProvider()
        await platform.start_live_session(session, audio)
    """

    def __init__(
        self,
        operating_mode: OperatingMode = OperatingMode.PROD_MODE,
        transcriber: Optional[ITranscriber] = None,
        risk_model: Optional[IRiskModel] = None
    ):
        """
        Initialize the PACING platform.

        Args:
            operating_mode: DEV_MODE (persist raw data) or PROD_MODE (ephemeral)
            transcriber: Optional transcriber (defaults to MockTranscriber)
            risk_model: Optional risk model (defaults to MockBayesianModel)
        """
        self.operating_mode = operating_mode
        self.session_stream = BasicSessionStream(operating_mode)

        # Default implementations (can be overridden via dependency injection)
        self._transcriber = transcriber
        self._risk_model = risk_model

        print(f"[PacingPlatform] Initialized in {operating_mode.value.upper()} mode")

        # Privacy reminder
        if operating_mode == OperatingMode.PROD_MODE:
            print("[PacingPlatform] PROD_MODE: Raw audio/transcripts will be ephemeral")
        else:
            print("[PacingPlatform] DEV_MODE: Raw audio/transcripts will be persisted")

    def set_transcriber(self, transcriber: ITranscriber) -> None:
        """
        Inject a custom transcriber.

        Args:
            transcriber: Transcriber implementation
        """
        self._transcriber = transcriber
        print(f"[PacingPlatform] Transcriber set: {transcriber.__class__.__name__}")

    def set_risk_model(self, model: IRiskModel) -> None:
        """
        Inject a custom risk model.

        Args:
            model: Risk model implementation
        """
        self._risk_model = model
        print(f"[PacingPlatform] Risk model set: {model.__class__.__name__}")

    def register_agent(self, agent: ISidecarAgent) -> None:
        """
        Register a sidecar agent.

        Args:
            agent: Sidecar agent implementation
        """
        self.session_stream.register_agent(agent)

    def unregister_agent(self, agent: ISidecarAgent) -> None:
        """
        Unregister a sidecar agent.

        Args:
            agent: Sidecar agent to remove
        """
        self.session_stream.unregister_agent(agent)

    async def start_live_session(
        self,
        session_metadata: SessionMetadata,
        audio_provider: IAudioProvider
    ) -> None:
        """
        Start a live clinical session.

        Args:
            session_metadata: Session information
            audio_provider: Audio source

        Raises:
            ValueError: If transcriber not configured
        """
        if not self._transcriber:
            # Use default mock transcriber
            from pacing.impl.defaults.mock_transcriber import MockTranscriber
            self._transcriber = MockTranscriber()
            print("[PacingPlatform] Using default MockTranscriber")

        await self.session_stream.start_session(
            session_metadata,
            audio_provider,
            self._transcriber
        )

    async def stop_live_session(self) -> None:
        """Stop the current live session."""
        await self.session_stream.stop_session()

    def calculate_risk(
        self,
        patient_graph: PatientGraph,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Calculate relapse risk for a patient (Longitudinal Mode).

        Args:
            patient_graph: Patient clinical history
            options: Optional model configuration

        Returns:
            RiskReport: Risk assessment

        Raises:
            ValueError: If risk model not configured
        """
        if not self._risk_model:
            # Use default mock model
            from pacing.impl.defaults.mock_risk_model import MockBayesianModel
            self._risk_model = MockBayesianModel()
            print("[PacingPlatform] Using default MockBayesianModel")

        return self._risk_model.calculate_risk(patient_graph, options)

    def get_simulation_context(
        self,
        patient_graph: PatientGraph
    ):
        """
        Create a simulation context for What-If analysis.

        Args:
            patient_graph: Patient clinical history

        Returns:
            SimulationContext: Context for running simulations

        Raises:
            ValueError: If risk model doesn't support simulation
        """
        if not self._risk_model:
            from pacing.impl.defaults.mock_risk_model import MockSimulationModel
            self._risk_model = MockSimulationModel()
            print("[PacingPlatform] Using default MockSimulationModel")

        if not isinstance(self._risk_model, ISimulationModel):
            raise ValueError(
                f"Risk model {self._risk_model.__class__.__name__} "
                "does not support simulation. Use a model that implements ISimulationModel."
            )

        from pacing.simulation.simulation_engine import SimulationContext
        return SimulationContext(patient_graph, self._risk_model)

    def get_platform_status(self) -> dict:
        """
        Get current platform status.

        Returns:
            dict: Platform configuration and status
        """
        return {
            "operating_mode": self.operating_mode.value,
            "session_active": self.session_stream.is_active,
            "transcriber": self._transcriber.__class__.__name__ if self._transcriber else None,
            "risk_model": self._risk_model.__class__.__name__ if self._risk_model else None,
            "registered_agents": [
                agent.get_agent_name()
                for agent in self.session_stream.registered_agents
            ]
        }
