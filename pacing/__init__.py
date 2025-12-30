"""
PACING: Platform for Agentic Clinical Intelligence with Networked Graphs

A clinical decision support system for Substance Use Disorder (SUD) treatment
with two main operational modes:

1. Live Session Mode: Real-time audio transcription with parallel agentic sidecars
2. Longitudinal Analysis Mode: Risk assessment and What-If scenario simulation

Key Features:
- Extensible plugin architecture with dependency injection
- Privacy-aware: DEV_MODE (persists data) vs PROD_MODE (ephemeral)
- Auditable: Human-in-the-loop verification with UncertaintyAuditor
- Explainable: Risk factors with evidence and contribution scores
"""

__version__ = "0.1.0"

# Primary platform interface
from pacing.platform import PacingPlatform, OperatingMode

# Core data models
from pacing.models.data_models import (
    TranscriptionResult,
    PatientGraph,
    Event,
    SubstanceUse,
    Intervention,
    RiskReport,
    SessionMetadata,
)

# Simulation engine
from pacing.simulation.simulation_engine import (
    SimulationContext,
    Mutation,
    MutationType,
    create_stable_housing_mutation,
    create_employment_mutation,
    create_mat_intervention_mutation,
)

# Default implementations (useful for getting started)
from pacing.impl.defaults.mock_transcriber import MockTranscriber, AdaptiveConfidenceTranscriber
from pacing.impl.defaults.mock_audio import MockAudioProvider
from pacing.impl.defaults.mock_risk_model import MockBayesianModel, MockSimulationModel

# Key agents
from pacing.impl.agents.uncertainty_auditor import UncertaintyAuditor

__all__ = [
    # Platform
    "PacingPlatform",
    "OperatingMode",
    # Data Models
    "TranscriptionResult",
    "PatientGraph",
    "Event",
    "SubstanceUse",
    "Intervention",
    "RiskReport",
    "SessionMetadata",
    # Simulation
    "SimulationContext",
    "Mutation",
    "MutationType",
    "create_stable_housing_mutation",
    "create_employment_mutation",
    "create_mat_intervention_mutation",
    # Mock Implementations
    "MockTranscriber",
    "AdaptiveConfidenceTranscriber",
    "MockAudioProvider",
    "MockBayesianModel",
    "MockSimulationModel",
    # Agents
    "UncertaintyAuditor",
]
