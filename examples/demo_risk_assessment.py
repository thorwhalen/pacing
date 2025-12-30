"""
Demonstration: Risk Assessment and What-If Simulation

This example shows how to:
1. Create a patient graph with clinical history
2. Calculate baseline relapse risk
3. Run What-If simulations to explore hypothetical scenarios
"""

from datetime import datetime, timedelta
from pacing import (
    PacingPlatform,
    PatientGraph,
    Event,
    EventType,
    SubstanceUse,
    SubstanceType,
    SubstanceUseStatus,
    Intervention,
    InterventionType,
    create_stable_housing_mutation,
    create_employment_mutation,
    create_mat_intervention_mutation,
)


def main():
    print("=" * 70)
    print("PACING Demo: Risk Assessment & What-If Simulation")
    print("=" * 70)

    # Initialize platform
    platform = PacingPlatform()

    # Create a patient graph with realistic clinical history
    patient = PatientGraph(
        patient_id="patient_demo_001",
        events=[
            Event(
                event_id="evt_001",
                event_type=EventType.JOB_CHANGE,
                description="Lost job due to company layoffs",
                date=datetime.now() - timedelta(days=15),
                impact_score=-0.7,  # Negative impact
            ),
            Event(
                event_id="evt_002",
                event_type=EventType.HOUSING_CHANGE,
                description="Currently in unstable housing situation",
                date=datetime.now() - timedelta(days=30),
                impact_score=-0.5,
            ),
        ],
        substance_use_records=[
            SubstanceUse(
                use_id="use_001",
                substance_type=SubstanceType.OPIOIDS,
                status=SubstanceUseStatus.REMISSION,
                date=datetime.now() - timedelta(days=180),
                severity=8,
                notes="6 months sober after completing inpatient treatment",
            )
        ],
        interventions=[
            Intervention(
                intervention_id="int_001",
                intervention_type=InterventionType.MEDICATION,
                description="Buprenorphine 8mg/naloxone 2mg daily",
                start_date=datetime.now() - timedelta(days=180),
                effectiveness_score=0.85,
            ),
            Intervention(
                intervention_id="int_002",
                intervention_type=InterventionType.THERAPY,
                description="Weekly cognitive-behavioral therapy sessions",
                start_date=datetime.now() - timedelta(days=150),
                effectiveness_score=0.75,
            ),
        ],
    )

    # Calculate baseline risk
    print("\n📊 BASELINE RISK ASSESSMENT")
    print("-" * 70)
    report = platform.calculate_risk(patient)

    print(f"\nPatient ID: {report.patient_id}")
    print(f"Risk Score: {report.risk_score:.1%}")
    print(f"Model Version: {report.model_version}")

    print("\n🔍 Contributing Risk Factors:")
    for i, factor in enumerate(report.risk_factors, 1):
        direction = "↑ INCREASES" if factor.contribution > 0 else "↓ DECREASES"
        print(f"\n  {i}. {factor.factor_name} {direction} risk by {abs(factor.contribution):.1%}")
        for evidence in factor.evidence:
            print(f"     • {evidence}")

    # Run What-If simulations
    print("\n\n" + "=" * 70)
    print("WHAT-IF SIMULATIONS")
    print("=" * 70)

    sim = platform.get_simulation_context(patient)

    # Scenario 1: Stable Housing
    print("\n🏠 Scenario 1: What if housing becomes stable?")
    print("-" * 70)
    result1 = sim.simulate_mutation(create_stable_housing_mutation())

    print(f"Baseline Risk:       {result1['baseline_risk']:.1%}")
    print(f"Modified Risk:       {result1['modified_risk']:.1%}")
    print(f"Risk Change:         {result1['delta']:+.1%}")
    print(f"\n{result1['explanation']}")

    # Scenario 2: Housing + Employment
    print("\n\n💼 Scenario 2: What if housing stabilizes AND patient gains employment?")
    print("-" * 70)
    result2 = sim.simulate_multiple_mutations(
        [create_stable_housing_mutation(), create_employment_mutation(employed=True)]
    )

    print(f"Baseline Risk:       {result2['baseline_risk']:.1%}")
    print(f"Modified Risk:       {result2['modified_risk']:.1%}")
    print(f"Risk Change:         {result2['delta']:+.1%}")
    print(f"\n{result2['explanation']}")

    # Scenario Comparison
    print("\n\n" + "=" * 70)
    print("SCENARIO COMPARISON")
    print("=" * 70)

    scenarios = {
        "A. Current State (Baseline)": [],
        "B. Stable Housing": [create_stable_housing_mutation()],
        "C. Housing + Employment": [
            create_stable_housing_mutation(),
            create_employment_mutation(employed=True),
        ],
        "D. Full Support (Housing + Employment + Additional MAT)": [
            create_stable_housing_mutation(),
            create_employment_mutation(employed=True),
            create_mat_intervention_mutation("naltrexone"),
        ],
    }

    comparison = sim.compare_scenarios(scenarios)

    print("\n📊 Risk Scores by Scenario (Ranked Best to Worst):\n")
    for i, scenario_name in enumerate(comparison["ranked"], 1):
        result = comparison["scenarios"][scenario_name]
        risk = result["modified_risk"]
        delta = result["delta"]

        # Visual indicator
        if risk < 0.3:
            indicator = "🟢 LOW"
        elif risk < 0.6:
            indicator = "🟡 MODERATE"
        else:
            indicator = "🔴 HIGH"

        print(f"  {i}. {scenario_name}")
        print(f"     Risk: {risk:.1%} {indicator}  (Change: {delta:+.1%})")

    print(f"\n✅ Recommended Scenario: {comparison['best_scenario']}")
    best = comparison["scenarios"][comparison["best_scenario"]]
    print(f"   Predicted Risk Reduction: {abs(best['delta']):.1%}")

    print("\n" + "=" * 70)
    print("Clinical Interpretation:")
    print("=" * 70)
    print(
        """
This analysis suggests that addressing social determinants of health
(housing stability, employment) in combination with continued medication-
assisted treatment could significantly reduce relapse risk.

⚕️  CLINICAL NOTE: These predictions are based on a simplified model and
    should be used to inform clinical judgment, not replace it. Always
    consider the full clinical context and patient preferences when
    making treatment decisions.
"""
    )


if __name__ == "__main__":
    main()
