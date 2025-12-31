"""Tests for data models."""

import pytest
from datetime import datetime
from pacing.models.data_models import (
    TranscriptionResult,
    ConfidenceLevel,
    PatientGraph,
    Event,
    EventType,
    SubstanceUse,
    SubstanceType,
    SubstanceUseStatus,
    Intervention,
    InterventionType,
    RiskReport,
    RiskFactor,
)


class TestTranscriptionResult:
    """Tests for TranscriptionResult model."""

    def test_transcription_result_creation(self):
        """Test creating a basic transcription result."""
        result = TranscriptionResult(
            text="Hello world",
            confidence_score=0.9,
        )
        assert result.text == "Hello world"
        assert result.confidence_score == 0.9
        assert result.is_partial is False
        assert result.timestamp is not None

    def test_confidence_level_high(self):
        """Test high confidence categorization."""
        result = TranscriptionResult(
            text="Test",
            confidence_score=0.9,
        )
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_confidence_level_medium(self):
        """Test medium confidence categorization."""
        result = TranscriptionResult(
            text="Test",
            confidence_score=0.75,
        )
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_confidence_level_low(self):
        """Test low confidence categorization."""
        result = TranscriptionResult(
            text="Test",
            confidence_score=0.5,
        )
        assert result.confidence_level == ConfidenceLevel.LOW

    def test_invalid_confidence_score(self):
        """Test that invalid confidence scores are rejected."""
        with pytest.raises(ValueError):
            TranscriptionResult(
                text="Test",
                confidence_score=1.5,  # Invalid: > 1.0
            )


class TestPatientGraph:
    """Tests for PatientGraph model."""

    def test_empty_patient_graph(self):
        """Test creating an empty patient graph."""
        graph = PatientGraph(patient_id="patient-123")
        assert graph.patient_id == "patient-123"
        assert len(graph.events) == 0
        assert len(graph.substance_use_records) == 0
        assert len(graph.interventions) == 0
        assert len(graph.edges) == 0

    def test_get_all_node_ids_empty(self):
        """Test getting node IDs from empty graph."""
        graph = PatientGraph(patient_id="patient-123")
        assert graph.get_all_node_ids() == []

    def test_get_all_node_ids_with_nodes(self):
        """Test getting node IDs from populated graph."""
        graph = PatientGraph(
            patient_id="patient-123",
            events=[
                Event(
                    event_id="event-1",
                    event_type=EventType.JOB_CHANGE,
                    description="Got new job",
                    date=datetime.now(),
                )
            ],
            substance_use_records=[
                SubstanceUse(
                    use_id="use-1",
                    substance_type=SubstanceType.ALCOHOL,
                    status=SubstanceUseStatus.RECOVERY,
                    date=datetime.now(),
                )
            ],
            interventions=[
                Intervention(
                    intervention_id="intervention-1",
                    intervention_type=InterventionType.THERAPY,
                    description="CBT sessions",
                    start_date=datetime.now(),
                )
            ],
        )
        node_ids = graph.get_all_node_ids()
        assert "event-1" in node_ids
        assert "use-1" in node_ids
        assert "intervention-1" in node_ids
        assert len(node_ids) == 3


class TestEvent:
    """Tests for Event model."""

    def test_event_creation(self):
        """Test creating a basic event."""
        event = Event(
            event_id="event-1",
            event_type=EventType.HOUSING_CHANGE,
            description="Moved to stable housing",
            date=datetime.now(),
            impact_score=0.8,
        )
        assert event.event_id == "event-1"
        assert event.event_type == EventType.HOUSING_CHANGE
        assert event.impact_score == 0.8

    def test_event_without_impact_score(self):
        """Test creating event without impact score."""
        event = Event(
            event_id="event-1",
            event_type=EventType.OTHER,
            description="Some event",
            date=datetime.now(),
        )
        assert event.impact_score is None


class TestSubstanceUse:
    """Tests for SubstanceUse model."""

    def test_substance_use_creation(self):
        """Test creating a substance use record."""
        use = SubstanceUse(
            use_id="use-1",
            substance_type=SubstanceType.OPIOIDS,
            status=SubstanceUseStatus.REMISSION,
            date=datetime.now(),
            severity=3,
        )
        assert use.use_id == "use-1"
        assert use.substance_type == SubstanceType.OPIOIDS
        assert use.status == SubstanceUseStatus.REMISSION
        assert use.severity == 3


class TestRiskReport:
    """Tests for RiskReport model."""

    def test_risk_report_creation(self):
        """Test creating a risk report."""
        report = RiskReport(
            patient_id="patient-123",
            risk_score=0.65,
            risk_factors=[
                RiskFactor(
                    factor_name="Recent Relapse",
                    contribution=0.30,
                    evidence=["Used substances 10 days ago"],
                )
            ],
        )
        assert report.patient_id == "patient-123"
        assert report.risk_score == 0.65
        assert len(report.risk_factors) == 1
        assert report.risk_factors[0].factor_name == "Recent Relapse"

    def test_risk_report_invalid_score(self):
        """Test that invalid risk scores are rejected."""
        with pytest.raises(ValueError):
            RiskReport(
                patient_id="patient-123",
                risk_score=1.5,  # Invalid: > 1.0
            )
