"""
Mock risk models for testing and demonstration.

These implementations provide deterministic risk assessments based on simple
rules, without requiring complex probabilistic inference.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from pacing.core.model_interfaces import IRiskModel, ISimulationModel
from pacing.models.data_models import (
    PatientGraph, RiskReport, RiskFactor,
    SubstanceUseStatus, EventType
)


class MockBayesianModel(IRiskModel):
    """
    Mock risk model using simple rule-based logic.

    This implementation demonstrates the IRiskModel interface without requiring
    actual Bayesian network inference. It uses heuristics like:
    - Recent substance use increases risk
    - Negative life events increase risk
    - Active interventions decrease risk
    - Longer sobriety decreases risk

    Risk Score Calculation:
    1. Start with base risk (0.50)
    2. Adjust based on recent substance use
    3. Adjust based on recent negative events
    4. Adjust based on active interventions
    5. Clamp to [0.0, 1.0]
    """

    def __init__(self, base_risk: float = 0.50):
        """
        Initialize the mock model.

        Args:
            base_risk: Starting risk score (0.0-1.0)
        """
        self.base_risk = base_risk

    def calculate_risk(
        self,
        patient_data: PatientGraph,
        options: Optional[Dict[str, Any]] = None
    ) -> RiskReport:
        """
        Calculate risk using simple heuristics.

        Args:
            patient_data: Patient clinical history
            options: Optional configuration

        Returns:
            RiskReport: Risk assessment with contributing factors
        """
        self.validate_input(patient_data)

        risk_score = self.base_risk
        factors = []

        # Factor 1: Recent substance use
        recent_use = self._check_recent_substance_use(patient_data)
        if recent_use:
            risk_score += 0.25
            factors.append(RiskFactor(
                factor_name="Recent Substance Use",
                contribution=0.25,
                evidence=[f"Active use or relapse in past 30 days"]
            ))

        # Factor 2: Negative life events
        negative_events = self._check_negative_events(patient_data)
        if negative_events:
            risk_score += 0.15
            factors.append(RiskFactor(
                factor_name="Recent Negative Life Events",
                contribution=0.15,
                evidence=[f"{len(negative_events)} stressful events in past 90 days"]
            ))

        # Factor 3: Active interventions (protective)
        active_interventions = self._check_active_interventions(patient_data)
        if active_interventions:
            risk_score -= 0.20
            factors.append(RiskFactor(
                factor_name="Active Treatment",
                contribution=-0.20,  # Negative = protective
                evidence=[f"{len(active_interventions)} active interventions"]
            ))

        # Factor 4: Sobriety duration (protective)
        days_sober = self._calculate_sobriety_days(patient_data)
        if days_sober > 180:  # 6+ months
            risk_score -= 0.15
            factors.append(RiskFactor(
                factor_name="Extended Sobriety",
                contribution=-0.15,
                evidence=[f"{days_sober} days since last use"]
            ))

        # Clamp risk to valid range
        risk_score = max(0.0, min(1.0, risk_score))

        # Sort factors by absolute contribution
        factors.sort(key=lambda f: abs(f.contribution), reverse=True)

        return RiskReport(
            patient_id=patient_data.patient_id,
            risk_score=risk_score,
            risk_factors=factors,
            model_version=self.get_model_version()
        )

    def _check_recent_substance_use(self, patient_data: PatientGraph) -> bool:
        """Check for substance use in past 30 days."""
        cutoff = datetime.now() - timedelta(days=30)
        for record in patient_data.substance_use_records:
            if record.date >= cutoff:
                if record.status in [SubstanceUseStatus.ACTIVE_USE, SubstanceUseStatus.RELAPSE]:
                    return True
        return False

    def _check_negative_events(self, patient_data: PatientGraph) -> list:
        """Find negative life events in past 90 days."""
        cutoff = datetime.now() - timedelta(days=90)
        negative_types = {EventType.TRAUMA, EventType.JOB_CHANGE, EventType.LEGAL_EVENT}

        negative_events = []
        for event in patient_data.events:
            if event.date >= cutoff:
                if event.event_type in negative_types:
                    negative_events.append(event)
                elif event.impact_score and event.impact_score < -0.3:
                    negative_events.append(event)

        return negative_events

    def _check_active_interventions(self, patient_data: PatientGraph) -> list:
        """Find currently active interventions."""
        now = datetime.now()
        active = []
        for intervention in patient_data.interventions:
            if intervention.start_date <= now:
                if intervention.end_date is None or intervention.end_date >= now:
                    active.append(intervention)
        return active

    def _calculate_sobriety_days(self, patient_data: PatientGraph) -> int:
        """Calculate days since last substance use."""
        if not patient_data.substance_use_records:
            return 999  # No record = assume long sobriety

        # Find most recent use
        recent_use = max(
            patient_data.substance_use_records,
            key=lambda r: r.date
        )

        if recent_use.status in [SubstanceUseStatus.RECOVERY, SubstanceUseStatus.REMISSION]:
            return (datetime.now() - recent_use.date).days
        else:
            return 0  # Active use

    def get_model_version(self) -> str:
        """Get model version."""
        return "mock-rule-based-v1.0"

    def get_model_metadata(self) -> dict:
        """Get model metadata."""
        return {
            "name": "MockBayesianModel",
            "version": self.get_model_version(),
            "type": "rule-based",
            "factors": [
                "recent_substance_use",
                "negative_life_events",
                "active_interventions",
                "sobriety_duration"
            ]
        }


class MockSimulationModel(MockBayesianModel, ISimulationModel):
    """
    Mock model that supports What-If simulation.

    Extends MockBayesianModel to enable scenario comparison.
    """

    def calculate_risk_delta(
        self,
        baseline: PatientGraph,
        modified: PatientGraph,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate risk change between baseline and modified scenarios.

        Args:
            baseline: Current patient state
            modified: Hypothetical patient state
            options: Optional configuration

        Returns:
            dict: Risk comparison with delta and explanation
        """
        baseline_report = self.calculate_risk(baseline, options)
        modified_report = self.calculate_risk(modified, options)

        delta = modified_report.risk_score - baseline_report.risk_score

        # Identify which factors changed
        changed_factors = []
        baseline_factor_names = {f.factor_name for f in baseline_report.risk_factors}
        modified_factor_names = {f.factor_name for f in modified_report.risk_factors}

        new_factors = modified_factor_names - baseline_factor_names
        removed_factors = baseline_factor_names - modified_factor_names

        if new_factors:
            changed_factors.append(f"New factors: {', '.join(new_factors)}")
        if removed_factors:
            changed_factors.append(f"Removed factors: {', '.join(removed_factors)}")

        return {
            "baseline_risk": baseline_report.risk_score,
            "modified_risk": modified_report.risk_score,
            "delta": delta,
            "delta_percent": delta * 100,
            "explanation": (
                f"Risk {'increased' if delta > 0 else 'decreased'} by {abs(delta):.2%}. "
                + (" ".join(changed_factors) if changed_factors else "Factor contributions changed.")
            ),
            "baseline_report": baseline_report,
            "modified_report": modified_report
        }
