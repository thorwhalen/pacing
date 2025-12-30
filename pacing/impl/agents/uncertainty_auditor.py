"""
Uncertainty Auditor Agent for the PACING platform.

This agent monitors transcription confidence scores and flags low-confidence
segments for human verification, implementing a human-in-the-loop safety mechanism.
"""

import uuid
from typing import Optional, List
from datetime import datetime

from pacing.core.agent_interfaces import ISidecarAgent
from pacing.models.data_models import TranscriptionResult, ReviewQueueItem


class UncertaintyAuditor(ISidecarAgent):
    """
    Sidecar agent that audits transcription confidence and flags uncertain segments.

    The UncertaintyAuditor implements a critical safety mechanism in clinical
    decision support: it identifies transcription segments where the acoustic
    model has low confidence and queues them for human verification.

    Design Rationale:
    - Clinical decisions should never be based on uncertain transcriptions
    - Medical/legal terminology often has low acoustic confidence
    - Human verification creates an auditable trail
    - This pattern enables "glass box" AI (explainable and verifiable)

    Configuration:
    - confidence_threshold: Below this, segment is flagged (default: 0.70)
    - priority_rules: Customize prioritization logic
    - max_queue_size: Maximum review queue size before warning

    Privacy Note:
    - The auditor stores transcriptions (text) for review, not raw audio
    - In production, implement retention policies for the review queue
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        max_queue_size: int = 100,
        auto_flag_medical_terms: bool = True
    ):
        """
        Initialize the Uncertainty Auditor.

        Args:
            confidence_threshold: Confidence below this triggers flagging (0.0-1.0)
            max_queue_size: Maximum items in review queue before warning
            auto_flag_medical_terms: Automatically flag medical/substance terms
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        self.confidence_threshold = confidence_threshold
        self.max_queue_size = max_queue_size
        self.auto_flag_medical_terms = auto_flag_medical_terms

        self.review_queue: List[ReviewQueueItem] = []
        self.total_transcriptions_processed = 0
        self.total_flagged = 0
        self.current_session_id: Optional[str] = None

        # Medical terms that should be flagged even with decent confidence
        self.medical_terms = {
            "buprenorphine", "naloxone", "methadone", "suboxone",
            "opioid", "benzodiazepine", "fentanyl", "morphine",
            "mg", "milligram", "dose", "dosage", "prescription"
        }

    async def on_transcription_update(
        self,
        transcription: TranscriptionResult,
        context: Optional[dict] = None
    ) -> None:
        """
        Process a transcription and flag if confidence is low.

        Args:
            transcription: The transcription to audit
            context: Optional context (session_id, patient_id, etc.)
        """
        self.total_transcriptions_processed += 1

        # Skip empty transcriptions
        if not transcription.text or not transcription.text.strip():
            return

        # Check if this segment should be flagged
        should_flag = False
        reason = ""
        priority = 1

        # Primary criterion: low confidence
        if transcription.confidence_score < self.confidence_threshold:
            should_flag = True
            reason = f"Low confidence score: {transcription.confidence_score:.2%}"
            # Higher priority for very low confidence
            if transcription.confidence_score < 0.50:
                priority = 5
            elif transcription.confidence_score < 0.60:
                priority = 3
            else:
                priority = 2

        # Secondary criterion: medical terms (even if confidence is acceptable)
        if self.auto_flag_medical_terms and not should_flag:
            text_lower = transcription.text.lower()
            found_terms = [term for term in self.medical_terms if term in text_lower]
            if found_terms:
                should_flag = True
                reason = f"Contains medical terms: {', '.join(found_terms)}"
                priority = 4  # High priority for medication names

        # If flagged, add to review queue
        if should_flag:
            self._add_to_review_queue(transcription, reason, priority, context)

    def _add_to_review_queue(
        self,
        transcription: TranscriptionResult,
        reason: str,
        priority: int,
        context: Optional[dict] = None
    ) -> None:
        """
        Add an item to the review queue.

        Args:
            transcription: The transcription to flag
            reason: Why it was flagged
            priority: Priority level (1-5)
            context: Additional context
        """
        item = ReviewQueueItem(
            item_id=str(uuid.uuid4()),
            transcription=transcription,
            reason=reason,
            priority=priority
        )

        self.review_queue.append(item)
        self.total_flagged += 1

        # Sort queue by priority (highest first)
        self.review_queue.sort(key=lambda x: x.priority, reverse=True)

        # Warn if queue is getting large
        if len(self.review_queue) > self.max_queue_size:
            print(
                f"[UncertaintyAuditor] WARNING: Review queue size ({len(self.review_queue)}) "
                f"exceeds max ({self.max_queue_size}). Consider reviewing items."
            )

        # Log for demonstration
        print(
            f"[UncertaintyAuditor] Flagged: \"{transcription.text}\" "
            f"[Priority: {priority}, Reason: {reason}]"
        )

    def on_session_start(self, session_id: str, metadata: dict) -> None:
        """
        Initialize for a new session.

        Args:
            session_id: Session identifier
            metadata: Session metadata
        """
        self.current_session_id = session_id
        # Note: We do NOT clear the review queue here, as items may span sessions
        # In production, implement session-specific queues
        print(f"[UncertaintyAuditor] Session started: {session_id}")

    def on_session_end(self, session_id: str) -> None:
        """
        Finalize processing for a session.

        Args:
            session_id: Session identifier
        """
        print(
            f"[UncertaintyAuditor] Session ended: {session_id}. "
            f"Processed {self.total_transcriptions_processed} transcriptions, "
            f"flagged {self.total_flagged} ({self.total_flagged/max(1, self.total_transcriptions_processed)*100:.1f}%)."
        )

        # Reset counters for next session
        self.total_transcriptions_processed = 0
        self.total_flagged = 0
        self.current_session_id = None

    def get_agent_name(self) -> str:
        """Get the agent name."""
        return "UncertaintyAuditor"

    def get_agent_status(self) -> dict:
        """Get the current status of the auditor."""
        return {
            "agent_name": self.get_agent_name(),
            "status": "active",
            "confidence_threshold": self.confidence_threshold,
            "review_queue_size": len(self.review_queue),
            "total_processed": self.total_transcriptions_processed,
            "total_flagged": self.total_flagged,
            "current_session": self.current_session_id
        }

    def get_review_queue(self) -> List[ReviewQueueItem]:
        """
        Get all items in the review queue.

        Returns:
            List[ReviewQueueItem]: Items sorted by priority (highest first)
        """
        return self.review_queue.copy()

    def get_unreviewed_items(self) -> List[ReviewQueueItem]:
        """
        Get only unreviewed items from the queue.

        Returns:
            List[ReviewQueueItem]: Unreviewed items
        """
        return [item for item in self.review_queue if not item.reviewed]

    def mark_reviewed(
        self,
        item_id: str,
        reviewer_notes: Optional[str] = None
    ) -> bool:
        """
        Mark an item as reviewed.

        Args:
            item_id: The item ID to mark as reviewed
            reviewer_notes: Optional notes from the reviewer

        Returns:
            bool: True if item was found and marked, False otherwise
        """
        for item in self.review_queue:
            if item.item_id == item_id:
                item.reviewed = True
                item.reviewer_notes = reviewer_notes
                print(f"[UncertaintyAuditor] Item {item_id} marked as reviewed")
                return True

        return False

    def clear_reviewed_items(self) -> int:
        """
        Remove reviewed items from the queue.

        Returns:
            int: Number of items removed
        """
        initial_size = len(self.review_queue)
        self.review_queue = [item for item in self.review_queue if not item.reviewed]
        removed = initial_size - len(self.review_queue)

        if removed > 0:
            print(f"[UncertaintyAuditor] Cleared {removed} reviewed items from queue")

        return removed

    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics about auditor performance.

        Returns:
            dict: Statistics including flagging rates, priority distribution, etc.
        """
        unreviewed = self.get_unreviewed_items()

        priority_counts = {}
        for item in self.review_queue:
            priority_counts[item.priority] = priority_counts.get(item.priority, 0) + 1

        return {
            "total_in_queue": len(self.review_queue),
            "unreviewed": len(unreviewed),
            "reviewed": len(self.review_queue) - len(unreviewed),
            "priority_distribution": priority_counts,
            "session_stats": {
                "transcriptions_processed": self.total_transcriptions_processed,
                "transcriptions_flagged": self.total_flagged,
                "flagging_rate": (
                    self.total_flagged / max(1, self.total_transcriptions_processed)
                )
            }
        }
