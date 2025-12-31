"""Tests for mock risk models."""

import pytest
from datetime import datetime, timedelta
from pacing.impl.defaults.mock_risk_model import MockBayesianModel, MockSimulationModel
from pacing.models.data_models import (
    PatientGraph,
    Event,
    EventType,
    SubstanceUse,
    SubstanceType,
    SubstanceUseStatus,
    Intervention,
    InterventionType,
)


class TestMockBayesianModel:
    """Tests for MockBayesianModel."""

    def test_model_creation(self):
        """Test creating a model with default base risk."""
        model = MockBayesianModel()
        assert model.base_risk == 0.50

    def test_model_creation_with_custom_base_risk(self):
        """Test creating a model with custom base risk."""
        model = MockBayesianModel(base_risk=0.30)
        assert model.base_risk == 0.30

    def test_model_version(self):
        """Test getting model version."""
        model = MockBayesianModel()
        assert "mock" in model.get_model_version().lower()

    def test_empty_patient_graph(self):
        """Test risk calculation for empty patient graph."""
        model = MockBayesianModel(base_risk=0.50)
        graph = PatientGraph(patient_id="patient-123")

        report = model.calculate_risk(graph)

        assert report.patient_id == "patient-123"
        # Empty graph should give base risk (possibly adjusted for no records)
        assert 0.0 <= report.risk_score <= 1.0

    def test_recent_substance_use_increases_risk(self):
        """Test that recent substance use increases risk."""
        model = MockBayesianModel(base_risk=0.30)

        # Create graph with recent substance use
        graph = PatientGraph(
            patient_id="patient-123",
            substance_use_records=[
                SubstanceUse(
                    use_id="use-1",
                    substance_type=SubstanceType.ALCOHOL,
                    status=SubstanceUseStatus.RELAPSE,
                    date=datetime.now() - timedelta(days=5),  # Recent
                )
            ],
        )

        report = model.calculate_risk(graph)

        # Should be higher than base risk
        assert report.risk_score > model.base_risk
        # Should have a factor for recent substance use
        factor_names = [f.factor_name for f in report.risk_factors]
        assert any("substance" in name.lower() for name in factor_names)

    def test_old_substance_use_does_not_increase_risk(self):
        """Test that old substance use doesn't trigger recent use factor."""
        model = MockBayesianModel(base_risk=0.30)

        # Create graph with old substance use
        graph = PatientGraph(
            patient_id="patient-123",
            substance_use_records=[
                SubstanceUse(
                    use_id="use-1",
                    substance_type=SubstanceType.ALCOHOL,
                    status=SubstanceUseStatus.RECOVERY,
                    date=datetime.now() - timedelta(days=200),  # Old
                )
            ],
        )

        report = model.calculate_risk(graph)

        # Should not have recent substance use factor
        factor_names = [f.factor_name for f in report.risk_factors]
        assert not any("recent substance" in name.lower() for name in factor_names)

    def test_active_interventions_decrease_risk(self):
        """Test that active interventions decrease risk."""
        model = MockBayesianModel(base_risk=0.50)

        # Create graph with active intervention
        graph = PatientGraph(
            patient_id="patient-123",
            interventions=[
                Intervention(
                    intervention_id="int-1",
                    intervention_type=InterventionType.THERAPY,
                    description="CBT",
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=None,  # Still active
                )
            ],
        )

        report = model.calculate_risk(graph)

        # Should be lower than base risk
        assert report.risk_score < model.base_risk
        # Should have a factor for active treatment
        factor_names = [f.factor_name for f in report.risk_factors]
        assert any("treatment" in name.lower() or "intervention" in name.lower() for name in factor_names)

    def test_negative_events_increase_risk(self):
        """Test that negative life events increase risk."""
        model = MockBayesianModel(base_risk=0.30)

        # Create graph with recent negative event and some substance use to prevent sobriety bonus
        graph = PatientGraph(
            patient_id="patient-123",
            events=[
                Event(
                    event_id="event-1",
                    event_type=EventType.TRAUMA,
                    description="Traumatic event",
                    date=datetime.now() - timedelta(days=20),
                    impact_score=-0.8,
                )
            ],
            substance_use_records=[
                SubstanceUse(
                    use_id="use-1",
                    substance_type=SubstanceType.ALCOHOL,
                    status=SubstanceUseStatus.RECOVERY,
                    date=datetime.now() - timedelta(days=100),  # Not recent, but prevents long sobriety bonus
                )
            ],
        )

        report = model.calculate_risk(graph)

        # Should be higher than base risk
        assert report.risk_score > model.base_risk
        # Should have a factor for negative events
        factor_names = [f.factor_name for f in report.risk_factors]
        assert any("event" in name.lower() for name in factor_names)

    def test_risk_score_clamped_to_valid_range(self):
        """Test that risk scores are clamped to [0.0, 1.0]."""
        model = MockBayesianModel(base_risk=0.90)

        # Create graph with many risk-increasing factors
        graph = PatientGraph(
            patient_id="patient-123",
            substance_use_records=[
                SubstanceUse(
                    use_id="use-1",
                    substance_type=SubstanceType.OPIOIDS,
                    status=SubstanceUseStatus.RELAPSE,
                    date=datetime.now() - timedelta(days=1),
                )
            ],
            events=[
                Event(
                    event_id="event-1",
                    event_type=EventType.TRAUMA,
                    description="Traumatic event",
                    date=datetime.now() - timedelta(days=10),
                    impact_score=-0.9,
                )
            ],
        )

        report = model.calculate_risk(graph)

        # Should be clamped to max 1.0
        assert report.risk_score <= 1.0
        assert report.risk_score >= 0.0

    def test_extended_sobriety_decreases_risk(self):
        """Test that extended sobriety decreases risk."""
        model = MockBayesianModel(base_risk=0.50)

        # Create graph with long sobriety period
        graph = PatientGraph(
            patient_id="patient-123",
            substance_use_records=[
                SubstanceUse(
                    use_id="use-1",
                    substance_type=SubstanceType.ALCOHOL,
                    status=SubstanceUseStatus.RECOVERY,
                    date=datetime.now() - timedelta(days=365),  # 1 year ago
                )
            ],
        )

        report = model.calculate_risk(graph)

        # Should be lower than base risk
        assert report.risk_score < model.base_risk
        # Should have sobriety factor
        factor_names = [f.factor_name for f in report.risk_factors]
        assert any("sobriety" in name.lower() for name in factor_names)


class TestMockSimulationModel:
    """Tests for MockSimulationModel."""

    def test_simulation_model_creation(self):
        """Test creating a simulation model."""
        model = MockSimulationModel()
        assert model.base_risk == 0.50

    def test_calculate_risk_delta_no_change(self):
        """Test risk delta when graphs are identical."""
        model = MockSimulationModel()

        graph = PatientGraph(patient_id="patient-123")

        result = model.calculate_risk_delta(graph, graph)

        assert result["delta"] == 0.0
        assert result["baseline_risk"] == result["modified_risk"]

    def test_calculate_risk_delta_with_intervention(self):
        """Test risk delta when adding an intervention."""
        model = MockSimulationModel()

        baseline = PatientGraph(patient_id="patient-123")

        modified = PatientGraph(
            patient_id="patient-123",
            interventions=[
                Intervention(
                    intervention_id="int-1",
                    intervention_type=InterventionType.MEDICATION,
                    description="MAT",
                    start_date=datetime.now(),
                )
            ],
        )

        result = model.calculate_risk_delta(baseline, modified)

        # Modified should have lower risk (negative delta)
        assert result["delta"] < 0.0
        assert result["modified_risk"] < result["baseline_risk"]
        assert "explanation" in result

    def test_calculate_risk_delta_with_negative_event(self):
        """Test risk delta when adding a negative event."""
        model = MockSimulationModel()

        baseline = PatientGraph(patient_id="patient-123")

        modified = PatientGraph(
            patient_id="patient-123",
            events=[
                Event(
                    event_id="event-1",
                    event_type=EventType.TRAUMA,
                    description="Traumatic event",
                    date=datetime.now(),
                    impact_score=-0.7,
                )
            ],
        )

        result = model.calculate_risk_delta(baseline, modified)

        # Modified should have higher risk (positive delta)
        assert result["delta"] > 0.0
        assert result["modified_risk"] > result["baseline_risk"]
