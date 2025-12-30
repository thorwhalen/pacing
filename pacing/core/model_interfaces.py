"""
Risk model interfaces for the PACING platform.

These interfaces define how probabilistic models assess relapse risk based on
patient data. Models can range from simple rule-based systems to complex
Bayesian networks or machine learning models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from pacing.models.data_models import PatientGraph, RiskReport


class IRiskModel(ABC):
    """
    Abstract interface for relapse risk assessment models.

    Risk models analyze a patient's clinical history (represented as a PatientGraph)
    and produce a RiskReport with:
    - Overall risk score (0.0 to 1.0)
    - Contributing risk factors ranked by importance
    - Optional confidence intervals

    Design Philosophy:
    - Models should be stateless (pure functions of input data)
    - They should explain their reasoning (list contributing factors)
    - They should be versionable (model_version in RiskReport)
    - They should be auditable (factors include evidence)

    Implementation Patterns:
    - Rule-based: Simple heuristics for demonstration/baseline
    - Bayesian Network: Probabilistic graphical models (pgmpy, PyMC)
    - Machine Learning: Trained models (scikit-learn, PyTorch)
    - Ensemble: Combine multiple models
    """

    @abstractmethod
    def calculate_risk(
        self,
        patient_data: PatientGraph,
        options: Optional[Dict[str, Any]] = None
    ) -> RiskReport:
        """
        Calculate relapse risk for a patient.

        Args:
            patient_data: The patient's clinical history as a graph
            options: Optional configuration (e.g., time horizon, focus areas)

        Returns:
            RiskReport: Risk assessment with score and contributing factors

        Example:
            model = MyBayesianModel()
            report = model.calculate_risk(patient_graph)
            print(f"Risk: {report.risk_score:.2%}")
            for factor in report.risk_factors:
                print(f"  - {factor.factor_name}: {factor.contribution:.2%}")
        """
        pass

    def get_model_version(self) -> str:
        """
        Get the version identifier for this model.

        Returns:
            str: Version string (e.g., "bayesian-v1.2", "ml-rf-20231201")

        Notes:
            - Version should change when model logic or parameters change
            - Used for auditability and reproducibility
        """
        return "unknown"

    def get_model_metadata(self) -> dict:
        """
        Get metadata about this model.

        Returns:
            dict: Model information (type, training data, features, etc.)
        """
        return {
            "name": self.__class__.__name__,
            "version": self.get_model_version(),
            "type": "unknown"
        }

    def validate_input(self, patient_data: PatientGraph) -> bool:
        """
        Validate that the patient data is sufficient for this model.

        Args:
            patient_data: Patient graph to validate

        Returns:
            bool: True if data is valid

        Raises:
            ValueError: If data is invalid (with explanation)
        """
        if not patient_data.patient_id:
            raise ValueError("Patient ID is required")
        return True


class ISimulationModel(IRiskModel):
    """
    Extended interface for models that support What-If simulation.

    Simulation models can handle hypothetical modifications to patient data
    and compare the resulting risk to a baseline.
    """

    @abstractmethod
    def calculate_risk_delta(
        self,
        baseline: PatientGraph,
        modified: PatientGraph,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate the change in risk between baseline and modified scenarios.

        Args:
            baseline: The current/actual patient state
            modified: The hypothetical patient state (e.g., "What if housing stable?")
            options: Optional configuration

        Returns:
            dict: Contains 'baseline_risk', 'modified_risk', 'delta', and 'explanation'

        Example:
            baseline_graph = load_patient_graph(patient_id)
            modified_graph = apply_intervention(baseline_graph, "stable_housing")

            delta = model.calculate_risk_delta(baseline_graph, modified_graph)
            print(f"Risk reduction: {delta['delta']:.2%}")
        """
        pass
