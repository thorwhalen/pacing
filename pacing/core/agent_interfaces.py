"""
Agent interfaces for the PACING platform.

Sidecar agents are parallel processes that subscribe to the transcription stream
and perform specialized tasks (guidance, auditing, scribe extraction, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any

from pacing.models.data_models import TranscriptionResult, ExtractedEntity


class ISidecarAgent(ABC):
    """
    Abstract interface for sidecar agents.

    Sidecar agents are independent processes that observe the transcription stream
    and perform specialized analysis in real-time. Examples include:

    - **Guide**: Monitors for missing information and prompts the clinician
    - **Auditor**: Flags low-confidence transcription segments for review
    - **Scribe**: Extracts structured entities (medications, events) from text

    Design Philosophy:
    - Agents should be independent and non-blocking
    - They should handle their own state and context
    - They communicate via well-defined outputs (e.g., ReviewQueueItems, ExtractedEntities)
    - Agents should be composable (multiple agents can run simultaneously)

    Privacy Considerations:
    - Agents receive transcriptions but not raw audio (separation of concerns)
    - Extracted data (entities) is retained; raw transcriptions may be ephemeral
    """

    @abstractmethod
    async def on_transcription_update(
        self,
        transcription: TranscriptionResult,
        context: Optional[dict] = None
    ) -> None:
        """
        Process a new transcription update.

        This is the main callback that agents implement. It is called whenever
        a new transcription (partial or final) becomes available.

        Args:
            transcription: The new transcription result
            context: Optional context (e.g., session metadata, patient ID)

        Notes:
            - This method should be non-blocking and fast
            - Long-running processing should be done asynchronously
            - Exceptions should be caught and logged, not propagated
        """
        pass

    def on_session_start(self, session_id: str, metadata: dict) -> None:
        """
        Called when a new clinical session begins.

        Use this to initialize agent state, clear buffers, etc.

        Args:
            session_id: Unique identifier for this session
            metadata: Session metadata (patient_id, clinician_id, etc.)
        """
        pass

    def on_session_end(self, session_id: str) -> None:
        """
        Called when a clinical session ends.

        Use this to finalize processing, flush buffers, generate summaries, etc.

        Args:
            session_id: Unique identifier for the ending session
        """
        pass

    @abstractmethod
    def get_agent_name(self) -> str:
        """
        Get the name of this agent.

        Returns:
            str: Human-readable agent name (e.g., "Guide", "Auditor", "Scribe")
        """
        pass

    def get_agent_status(self) -> dict:
        """
        Get the current status of this agent.

        Returns:
            dict: Status information (e.g., items processed, errors, queue size)
        """
        return {
            "agent_name": self.get_agent_name(),
            "status": "active"
        }


class IScribeAgent(ISidecarAgent):
    """
    Specialized interface for scribe agents that extract structured entities.

    Scribe agents parse transcriptions to identify and extract:
    - Medications mentioned
    - Life events discussed
    - Substance use reported
    - Interventions planned
    """

    @abstractmethod
    def get_extracted_entities(self) -> List[ExtractedEntity]:
        """
        Retrieve all entities extracted so far in this session.

        Returns:
            List[ExtractedEntity]: Extracted entities with confidence scores

        Notes:
            - This list accumulates during a session
            - Call this at session end to get the complete extraction
        """
        pass

    def clear_extracted_entities(self) -> None:
        """
        Clear the extracted entities buffer.

        Typically called at the start of a new session.
        """
        pass
