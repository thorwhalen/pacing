"""
Simulation engine for What-If analysis in PACING.

This module enables clinicians to explore hypothetical scenarios and understand
how changes in a patient's circumstances might affect relapse risk.
"""

import copy
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from enum import Enum

from pacing.core.model_interfaces import ISimulationModel
from pacing.models.data_models import (
    PatientGraph, Event, Intervention, SubstanceUse,
    EventType, InterventionType, SubstanceUseStatus
)


class MutationType(str, Enum):
    """Types of mutations that can be applied to a patient graph."""
    ADD_EVENT = "add_event"
    REMOVE_EVENT = "remove_event"
    MODIFY_EVENT = "modify_event"
    ADD_INTERVENTION = "add_intervention"
    REMOVE_INTERVENTION = "remove_intervention"
    MODIFY_INTERVENTION = "modify_intervention"
    ADD_SUBSTANCE_USE = "add_substance_use"
    MODIFY_HOUSING = "modify_housing"
    MODIFY_EMPLOYMENT = "modify_employment"


class Mutation:
    """
    A modification to apply to a patient graph for simulation.

    Mutations represent "What If" questions:
    - What if housing became stable?
    - What if the patient started medication-assisted treatment?
    - What if the patient experienced a traumatic event?
    """

    def __init__(
        self,
        mutation_type: MutationType,
        parameters: Dict[str, Any],
        description: Optional[str] = None
    ):
        """
        Initialize a mutation.

        Args:
            mutation_type: Type of mutation to apply
            parameters: Parameters specific to this mutation type
            description: Human-readable description (e.g., "Stable housing")
        """
        self.mutation_type = mutation_type
        self.parameters = parameters
        self.description = description or str(mutation_type)

    def apply(self, graph: PatientGraph) -> PatientGraph:
        """
        Apply this mutation to a patient graph.

        Args:
            graph: The graph to modify

        Returns:
            PatientGraph: A new graph with the mutation applied
        """
        # Create a deep copy to avoid modifying the original
        modified = copy.deepcopy(graph)

        if self.mutation_type == MutationType.ADD_EVENT:
            event = Event(**self.parameters)
            modified.events.append(event)

        elif self.mutation_type == MutationType.REMOVE_EVENT:
            event_id = self.parameters.get("event_id")
            modified.events = [e for e in modified.events if e.event_id != event_id]

        elif self.mutation_type == MutationType.MODIFY_EVENT:
            event_id = self.parameters.get("event_id")
            for event in modified.events:
                if event.event_id == event_id:
                    for key, value in self.parameters.items():
                        if key != "event_id" and hasattr(event, key):
                            setattr(event, key, value)

        elif self.mutation_type == MutationType.ADD_INTERVENTION:
            intervention = Intervention(**self.parameters)
            modified.interventions.append(intervention)

        elif self.mutation_type == MutationType.REMOVE_INTERVENTION:
            intervention_id = self.parameters.get("intervention_id")
            modified.interventions = [
                i for i in modified.interventions if i.intervention_id != intervention_id
            ]

        elif self.mutation_type == MutationType.MODIFY_HOUSING:
            # Add housing stability as a positive life event
            housing_event = Event(
                event_id=f"sim_housing_{datetime.now().timestamp()}",
                event_type=EventType.HOUSING_CHANGE,
                description=self.parameters.get("description", "Housing stabilized"),
                date=datetime.now(),
                impact_score=self.parameters.get("impact_score", 0.7)  # Positive impact
            )
            modified.events.append(housing_event)
            modified.metadata["simulated_housing_status"] = self.parameters.get("status", "stable")

        elif self.mutation_type == MutationType.MODIFY_EMPLOYMENT:
            # Add employment change as a life event
            employment_event = Event(
                event_id=f"sim_employment_{datetime.now().timestamp()}",
                event_type=EventType.JOB_CHANGE,
                description=self.parameters.get("description", "Employment status changed"),
                date=datetime.now(),
                impact_score=self.parameters.get("impact_score", 0.5)
            )
            modified.events.append(employment_event)
            modified.metadata["simulated_employment_status"] = self.parameters.get("status", "employed")

        return modified


class SimulationContext:
    """
    Context for running What-If simulations on patient data.

    The SimulationContext:
    1. Takes a baseline patient graph (current state)
    2. Applies hypothetical mutations (e.g., "What if housing stable?")
    3. Runs risk models on both baseline and modified graphs
    4. Compares the results to show the impact of the hypothetical change

    Example Usage:
        # Load patient data
        patient_graph = load_patient_data(patient_id)

        # Create simulation context
        model = MockSimulationModel()
        sim = SimulationContext(patient_graph, model)

        # Run "What If" scenario
        result = sim.simulate_mutation(
            Mutation(
                MutationType.MODIFY_HOUSING,
                {"status": "stable", "impact_score": 0.8},
                description="Housing becomes stable"
            )
        )

        print(f"Current risk: {result['baseline_risk']:.2%}")
        print(f"Predicted risk with stable housing: {result['modified_risk']:.2%}")
        print(f"Risk reduction: {abs(result['delta']):.2%}")

    Design Philosophy:
    - Simulations are non-destructive (original data never modified)
    - Multiple scenarios can be compared side-by-side
    - Results include explanations for interpretability
    - Suitable for clinical decision support and patient education
    """

    def __init__(self, baseline_graph: PatientGraph, model: ISimulationModel):
        """
        Initialize the simulation context.

        Args:
            baseline_graph: The current/actual patient state
            model: Risk model that supports simulation
        """
        self.baseline_graph = baseline_graph
        self.model = model
        self.simulation_history: List[Dict[str, Any]] = []

    def simulate_mutation(
        self,
        mutation: Mutation,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate the impact of a single mutation.

        Args:
            mutation: The hypothetical change to apply
            options: Optional model configuration

        Returns:
            dict: Simulation results with risk comparison
        """
        # Apply mutation to create hypothetical scenario
        modified_graph = mutation.apply(self.baseline_graph)

        # Calculate risk delta
        result = self.model.calculate_risk_delta(
            self.baseline_graph,
            modified_graph,
            options
        )

        # Add metadata
        result["mutation"] = {
            "type": mutation.mutation_type,
            "description": mutation.description,
            "parameters": mutation.parameters
        }
        result["timestamp"] = datetime.now()

        # Store in history
        self.simulation_history.append(result)

        return result

    def simulate_multiple_mutations(
        self,
        mutations: List[Mutation],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate the combined impact of multiple mutations.

        This allows exploring complex scenarios like:
        "What if housing stabilizes AND patient starts MAT AND gets employed?"

        Args:
            mutations: List of mutations to apply together
            options: Optional model configuration

        Returns:
            dict: Simulation results with risk comparison
        """
        # Apply all mutations sequentially
        modified_graph = self.baseline_graph
        for mutation in mutations:
            modified_graph = mutation.apply(modified_graph)

        # Calculate risk delta
        result = self.model.calculate_risk_delta(
            self.baseline_graph,
            modified_graph,
            options
        )

        # Add metadata
        result["mutations"] = [
            {
                "type": m.mutation_type,
                "description": m.description,
                "parameters": m.parameters
            }
            for m in mutations
        ]
        result["timestamp"] = datetime.now()

        # Store in history
        self.simulation_history.append(result)

        return result

    def compare_scenarios(
        self,
        scenarios: Dict[str, List[Mutation]],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple alternative scenarios side-by-side.

        Example:
            scenarios = {
                "Scenario A: MAT Only": [mat_mutation],
                "Scenario B: MAT + Housing": [mat_mutation, housing_mutation],
                "Scenario C: MAT + Housing + Employment": [mat_mutation, housing_mutation, job_mutation]
            }

            comparison = sim.compare_scenarios(scenarios)

        Args:
            scenarios: Dict mapping scenario names to mutation lists
            options: Optional model configuration

        Returns:
            dict: Results for each scenario
        """
        results = {}

        for scenario_name, mutations in scenarios.items():
            result = self.simulate_multiple_mutations(mutations, options)
            result["scenario_name"] = scenario_name
            results[scenario_name] = result

        # Sort scenarios by predicted risk (best to worst)
        sorted_scenarios = sorted(
            results.items(),
            key=lambda item: item[1]["modified_risk"]
        )

        return {
            "scenarios": results,
            "ranked": [name for name, _ in sorted_scenarios],
            "best_scenario": sorted_scenarios[0][0] if sorted_scenarios else None
        }

    def get_simulation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of all simulations run in this context.

        Returns:
            List[Dict[str, Any]]: List of simulation results
        """
        return self.simulation_history.copy()

    def reset_baseline(self, new_baseline: PatientGraph) -> None:
        """
        Update the baseline graph (e.g., when patient data changes).

        Args:
            new_baseline: The new baseline patient state
        """
        self.baseline_graph = new_baseline
        self.simulation_history = []  # Clear history since baseline changed


# Convenience functions for common What-If scenarios

def create_stable_housing_mutation() -> Mutation:
    """Create a mutation representing stable housing."""
    return Mutation(
        MutationType.MODIFY_HOUSING,
        {
            "status": "stable",
            "description": "Housing became stable (shelter or permanent residence)",
            "impact_score": 0.7
        },
        description="Stable Housing"
    )


def create_employment_mutation(employed: bool = True) -> Mutation:
    """Create a mutation representing employment status change."""
    return Mutation(
        MutationType.MODIFY_EMPLOYMENT,
        {
            "status": "employed" if employed else "unemployed",
            "description": f"Patient {'gained' if employed else 'lost'} employment",
            "impact_score": 0.6 if employed else -0.6
        },
        description="Employed" if employed else "Unemployed"
    )


def create_mat_intervention_mutation(
    medication: str = "buprenorphine"
) -> Mutation:
    """Create a mutation for starting medication-assisted treatment (MAT)."""
    return Mutation(
        MutationType.ADD_INTERVENTION,
        {
            "intervention_id": f"sim_mat_{datetime.now().timestamp()}",
            "intervention_type": InterventionType.MEDICATION,
            "description": f"Medication-Assisted Treatment ({medication})",
            "start_date": datetime.now(),
            "effectiveness_score": 0.75
        },
        description=f"Start MAT ({medication})"
    )
