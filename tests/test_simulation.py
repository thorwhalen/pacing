"""Tests for simulation engine."""

import pytest
from datetime import datetime
from pacing.simulation.simulation_engine import (
    Mutation,
    MutationType,
    SimulationContext,
    create_stable_housing_mutation,
    create_employment_mutation,
    create_mat_intervention_mutation,
)
from pacing.impl.defaults.mock_risk_model import MockSimulationModel
from pacing.models.data_models import (
    PatientGraph,
    Event,
    EventType,
    Intervention,
    InterventionType,
)


class TestMutation:
    """Tests for Mutation class."""

    def test_mutation_creation(self):
        """Test creating a basic mutation."""
        mutation = Mutation(
            mutation_type=MutationType.ADD_EVENT,
            parameters={
                "event_id": "event-1",
                "event_type": EventType.HOUSING_CHANGE,
                "description": "Moved to stable housing",
                "date": datetime.now(),
            },
            description="Housing stabilized",
        )
        assert mutation.mutation_type == MutationType.ADD_EVENT
        assert mutation.description == "Housing stabilized"

    def test_add_event_mutation(self):
        """Test applying ADD_EVENT mutation."""
        graph = PatientGraph(patient_id="patient-123")
        assert len(graph.events) == 0

        mutation = Mutation(
            mutation_type=MutationType.ADD_EVENT,
            parameters={
                "event_id": "event-1",
                "event_type": EventType.HOUSING_CHANGE,
                "description": "Moved",
                "date": datetime.now(),
            },
        )

        modified_graph = mutation.apply(graph)

        # Original graph unchanged
        assert len(graph.events) == 0
        # Modified graph has new event
        assert len(modified_graph.events) == 1
        assert modified_graph.events[0].event_id == "event-1"

    def test_remove_event_mutation(self):
        """Test applying REMOVE_EVENT mutation."""
        graph = PatientGraph(
            patient_id="patient-123",
            events=[
                Event(
                    event_id="event-1",
                    event_type=EventType.OTHER,
                    description="Test event",
                    date=datetime.now(),
                )
            ],
        )
        assert len(graph.events) == 1

        mutation = Mutation(
            mutation_type=MutationType.REMOVE_EVENT,
            parameters={"event_id": "event-1"},
        )

        modified_graph = mutation.apply(graph)

        # Original graph unchanged
        assert len(graph.events) == 1
        # Modified graph has event removed
        assert len(modified_graph.events) == 0

    def test_modify_event_mutation(self):
        """Test applying MODIFY_EVENT mutation."""
        graph = PatientGraph(
            patient_id="patient-123",
            events=[
                Event(
                    event_id="event-1",
                    event_type=EventType.OTHER,
                    description="Original description",
                    date=datetime.now(),
                    impact_score=0.0,
                )
            ],
        )

        mutation = Mutation(
            mutation_type=MutationType.MODIFY_EVENT,
            parameters={
                "event_id": "event-1",
                "description": "Updated description",
                "impact_score": 0.8,
            },
        )

        modified_graph = mutation.apply(graph)

        # Original unchanged
        assert graph.events[0].description == "Original description"
        assert graph.events[0].impact_score == 0.0
        # Modified has updates
        assert modified_graph.events[0].description == "Updated description"
        assert modified_graph.events[0].impact_score == 0.8

    def test_add_intervention_mutation(self):
        """Test applying ADD_INTERVENTION mutation."""
        graph = PatientGraph(patient_id="patient-123")
        assert len(graph.interventions) == 0

        mutation = Mutation(
            mutation_type=MutationType.ADD_INTERVENTION,
            parameters={
                "intervention_id": "int-1",
                "intervention_type": InterventionType.THERAPY,
                "description": "CBT",
                "start_date": datetime.now(),
            },
        )

        modified_graph = mutation.apply(graph)

        assert len(graph.interventions) == 0
        assert len(modified_graph.interventions) == 1
        assert modified_graph.interventions[0].intervention_id == "int-1"

    def test_modify_housing_mutation(self):
        """Test applying MODIFY_HOUSING mutation."""
        graph = PatientGraph(patient_id="patient-123")

        mutation = Mutation(
            mutation_type=MutationType.MODIFY_HOUSING,
            parameters={
                "status": "stable",
                "description": "Housing stabilized",
                "impact_score": 0.7,
            },
        )

        modified_graph = mutation.apply(graph)

        # Should add a housing change event
        assert len(modified_graph.events) == 1
        assert modified_graph.events[0].event_type == EventType.HOUSING_CHANGE
        # Should update metadata
        assert "simulated_housing_status" in modified_graph.metadata
        assert modified_graph.metadata["simulated_housing_status"] == "stable"

    def test_modify_employment_mutation(self):
        """Test applying MODIFY_EMPLOYMENT mutation."""
        graph = PatientGraph(patient_id="patient-123")

        mutation = Mutation(
            mutation_type=MutationType.MODIFY_EMPLOYMENT,
            parameters={
                "status": "employed",
                "description": "Got job",
                "impact_score": 0.6,
            },
        )

        modified_graph = mutation.apply(graph)

        # Should add a job change event
        assert len(modified_graph.events) == 1
        assert modified_graph.events[0].event_type == EventType.JOB_CHANGE
        # Should update metadata
        assert "simulated_employment_status" in modified_graph.metadata


class TestSimulationContext:
    """Tests for SimulationContext class."""

    def test_simulation_context_creation(self):
        """Test creating a simulation context."""
        graph = PatientGraph(patient_id="patient-123")
        model = MockSimulationModel()
        sim = SimulationContext(graph, model)

        assert sim.baseline_graph == graph
        assert sim.model == model
        assert len(sim.simulation_history) == 0

    def test_simulate_mutation(self):
        """Test simulating a single mutation."""
        graph = PatientGraph(patient_id="patient-123")
        model = MockSimulationModel()
        sim = SimulationContext(graph, model)

        mutation = Mutation(
            mutation_type=MutationType.ADD_INTERVENTION,
            parameters={
                "intervention_id": "int-1",
                "intervention_type": InterventionType.MEDICATION,
                "description": "MAT",
                "start_date": datetime.now(),
            },
            description="Start MAT",
        )

        result = sim.simulate_mutation(mutation)

        # Should have risk comparison
        assert "baseline_risk" in result
        assert "modified_risk" in result
        assert "delta" in result
        # Should have mutation metadata
        assert "mutation" in result
        assert result["mutation"]["description"] == "Start MAT"
        # Should be in history
        assert len(sim.simulation_history) == 1

    def test_simulate_multiple_mutations(self):
        """Test simulating multiple mutations together."""
        graph = PatientGraph(patient_id="patient-123")
        model = MockSimulationModel()
        sim = SimulationContext(graph, model)

        mutations = [
            Mutation(
                mutation_type=MutationType.MODIFY_HOUSING,
                parameters={"status": "stable", "impact_score": 0.7},
                description="Stable housing",
            ),
            Mutation(
                mutation_type=MutationType.MODIFY_EMPLOYMENT,
                parameters={"status": "employed", "impact_score": 0.6},
                description="Employment",
            ),
        ]

        result = sim.simulate_multiple_mutations(mutations)

        # Should have combined effect
        assert "baseline_risk" in result
        assert "modified_risk" in result
        # Should track multiple mutations
        assert "mutations" in result
        assert len(result["mutations"]) == 2

    def test_compare_scenarios(self):
        """Test comparing multiple scenarios."""
        graph = PatientGraph(patient_id="patient-123")
        model = MockSimulationModel()
        sim = SimulationContext(graph, model)

        scenarios = {
            "Scenario A": [
                Mutation(
                    mutation_type=MutationType.MODIFY_HOUSING,
                    parameters={"status": "stable", "impact_score": 0.7},
                )
            ],
            "Scenario B": [
                Mutation(
                    mutation_type=MutationType.ADD_INTERVENTION,
                    parameters={
                        "intervention_id": "int-1",
                        "intervention_type": InterventionType.MEDICATION,
                        "description": "MAT",
                        "start_date": datetime.now(),
                    },
                )
            ],
        }

        result = sim.compare_scenarios(scenarios)

        # Should have results for both scenarios
        assert "scenarios" in result
        assert "Scenario A" in result["scenarios"]
        assert "Scenario B" in result["scenarios"]
        # Should rank scenarios
        assert "ranked" in result
        assert "best_scenario" in result

    def test_reset_baseline(self):
        """Test resetting the baseline graph."""
        graph1 = PatientGraph(patient_id="patient-123")
        model = MockSimulationModel()
        sim = SimulationContext(graph1, model)

        # Run a simulation
        mutation = Mutation(
            mutation_type=MutationType.MODIFY_HOUSING,
            parameters={"status": "stable", "impact_score": 0.7},
        )
        sim.simulate_mutation(mutation)
        assert len(sim.simulation_history) == 1

        # Reset with new baseline
        graph2 = PatientGraph(patient_id="patient-456")
        sim.reset_baseline(graph2)

        # Should have new baseline and cleared history
        assert sim.baseline_graph.patient_id == "patient-456"
        assert len(sim.simulation_history) == 0


class TestConvenienceFunctions:
    """Tests for convenience mutation creation functions."""

    def test_create_stable_housing_mutation(self):
        """Test creating stable housing mutation."""
        mutation = create_stable_housing_mutation()
        assert mutation.mutation_type == MutationType.MODIFY_HOUSING
        assert mutation.parameters["status"] == "stable"

    def test_create_employment_mutation_employed(self):
        """Test creating employment mutation (employed)."""
        mutation = create_employment_mutation(employed=True)
        assert mutation.mutation_type == MutationType.MODIFY_EMPLOYMENT
        assert mutation.parameters["status"] == "employed"
        assert mutation.parameters["impact_score"] > 0

    def test_create_employment_mutation_unemployed(self):
        """Test creating employment mutation (unemployed)."""
        mutation = create_employment_mutation(employed=False)
        assert mutation.mutation_type == MutationType.MODIFY_EMPLOYMENT
        assert mutation.parameters["status"] == "unemployed"
        assert mutation.parameters["impact_score"] < 0

    def test_create_mat_intervention_mutation(self):
        """Test creating MAT intervention mutation."""
        mutation = create_mat_intervention_mutation()
        assert mutation.mutation_type == MutationType.ADD_INTERVENTION
        assert "intervention_type" in mutation.parameters
        assert mutation.parameters["intervention_type"] == InterventionType.MEDICATION
