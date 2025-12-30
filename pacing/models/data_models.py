"""
Core data models for the PACING platform.

These Pydantic models define the structure of data flowing through the system,
ensuring type safety and validation at runtime.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ConfidenceLevel(str, Enum):
    """Transcription confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TranscriptionResult(BaseModel):
    """
    Result from a transcription operation.

    Attributes:
        text: The transcribed text
        timestamp: When this transcription was generated
        confidence_score: Acoustic confidence (0.0-1.0)
        speaker_id: Optional speaker identification
        is_partial: Whether this is a partial (streaming) result
    """
    text: str
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(ge=0.0, le=1.0)
    speaker_id: Optional[str] = None
    is_partial: bool = False

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Categorize confidence score into levels."""
        if self.confidence_score >= 0.85:
            return ConfidenceLevel.HIGH
        elif self.confidence_score >= 0.70:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class EventType(str, Enum):
    """Types of life events tracked in the patient graph."""
    JOB_CHANGE = "job_change"
    HOUSING_CHANGE = "housing_change"
    RELATIONSHIP_CHANGE = "relationship_change"
    TRAUMA = "trauma"
    LEGAL_EVENT = "legal_event"
    MEDICAL_EVENT = "medical_event"
    OTHER = "other"


class Event(BaseModel):
    """
    A life event in the patient's history.

    Events represent significant occurrences that may impact relapse risk.
    """
    event_id: str
    event_type: EventType
    description: str
    date: datetime
    impact_score: Optional[float] = Field(None, ge=-1.0, le=1.0,
                                          description="Estimated impact on recovery (-1=very negative, 1=very positive)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubstanceType(str, Enum):
    """Types of substances tracked."""
    ALCOHOL = "alcohol"
    OPIOIDS = "opioids"
    STIMULANTS = "stimulants"
    CANNABIS = "cannabis"
    BENZODIAZEPINES = "benzodiazepines"
    OTHER = "other"


class SubstanceUseStatus(str, Enum):
    """Status of substance use."""
    ACTIVE_USE = "active_use"
    RELAPSE = "relapse"
    REMISSION = "remission"
    RECOVERY = "recovery"


class SubstanceUse(BaseModel):
    """
    A substance use event or status change.
    """
    use_id: str
    substance_type: SubstanceType
    status: SubstanceUseStatus
    date: datetime
    severity: Optional[int] = Field(None, ge=1, le=10)
    notes: Optional[str] = None


class InterventionType(str, Enum):
    """Types of clinical interventions."""
    MEDICATION = "medication"
    THERAPY = "therapy"
    SUPPORT_GROUP = "support_group"
    HOSPITALIZATION = "hospitalization"
    OTHER = "other"


class Intervention(BaseModel):
    """
    A clinical intervention in the patient's treatment plan.
    """
    intervention_id: str
    intervention_type: InterventionType
    description: str
    start_date: datetime
    end_date: Optional[datetime] = None
    effectiveness_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class GraphEdge(BaseModel):
    """
    An edge connecting nodes in the patient graph.

    Edges can represent temporal or causal relationships.
    """
    source_id: str
    target_id: str
    relationship_type: str  # e.g., "leads_to", "causes", "precedes"
    weight: float = 1.0


class PatientGraph(BaseModel):
    """
    A structured representation of a patient's clinical history.

    This graph contains nodes (events, substance use, interventions) and edges
    (temporal/causal relationships) that are used for longitudinal reasoning.
    """
    patient_id: str
    events: List[Event] = Field(default_factory=list)
    substance_use_records: List[SubstanceUse] = Field(default_factory=list)
    interventions: List[Intervention] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_all_node_ids(self) -> List[str]:
        """Get all node IDs in the graph."""
        return (
            [e.event_id for e in self.events] +
            [s.use_id for s in self.substance_use_records] +
            [i.intervention_id for i in self.interventions]
        )


class RiskFactor(BaseModel):
    """
    A contributing factor to relapse risk.
    """
    factor_name: str
    contribution: float = Field(ge=0.0, le=1.0,
                               description="How much this factor contributes to overall risk")
    evidence: List[str] = Field(default_factory=list,
                               description="Supporting evidence from patient data")


class RiskReport(BaseModel):
    """
    Output from a risk assessment model.

    Attributes:
        patient_id: Patient identifier
        risk_score: Overall relapse risk probability (0.0-1.0)
        risk_factors: Contributing factors ranked by importance
        timestamp: When this assessment was generated
        model_version: Identifier for the model that generated this report
    """
    patient_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str = "default"
    confidence_interval: Optional[tuple[float, float]] = None


class ReviewQueueItem(BaseModel):
    """
    An item flagged for human review.

    Used by the Auditor agent to track low-confidence transcription segments
    that require verification.
    """
    item_id: str
    transcription: TranscriptionResult
    flagged_at: datetime = Field(default_factory=datetime.now)
    reason: str
    priority: int = Field(default=1, ge=1, le=5, description="1=lowest, 5=highest")
    reviewed: bool = False
    reviewer_notes: Optional[str] = None


class ExtractedEntity(BaseModel):
    """
    A structured entity extracted from transcription by the Scribe agent.

    Examples: medications mentioned, life events discussed, substance use reported.
    """
    entity_type: str  # "medication", "life_event", "substance_use", etc.
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_text: str  # Original text from which this was extracted
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionMetadata(BaseModel):
    """
    Metadata for a clinical session.
    """
    session_id: str
    patient_id: str
    clinician_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    session_type: str = "counseling"  # counseling, intake, follow-up, etc.
