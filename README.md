# PACING: Platform for Agentic Clinical Intelligence with Networked Graphs

A clinical decision support system for Substance Use Disorder (SUD) treatment that combines real-time AI assistance with longitudinal risk analysis.

## 🎯 Overview

PACING is a **scaffold-first platform** designed for extensibility, auditability, and privacy. It operates in two distinct modes:

### 1. Live Session Mode ("The Stream")
Real-time audio transcription with parallel **Agentic Sidecars** that assist clinicians:
- **Guide**: Monitors for missing critical information
- **Auditor**: Flags low-confidence transcriptions for human review
- **Scribe**: Extracts structured entities (medications, events) from conversations

### 2. Longitudinal Analysis Mode ("The Graph")
Patient history visualization and risk modeling:
- **Risk Assessment**: Probabilistic models estimate relapse risk
- **What-If Simulation**: Explore hypothetical scenarios ("What if housing stabilizes?")
- **Evidence-Based**: Models explain their reasoning with contributing factors

---

## 🏗️ Architecture Philosophy

### Glass Box Design
Unlike black-box AI, PACING is **auditable** and **explainable**:
- Risk models list contributing factors with evidence
- Low-confidence transcriptions are flagged for human review
- All decisions are traceable and verifiable

### Privacy by Design
Two operating modes enforce privacy boundaries:
- **`DEV_MODE`**: Raw audio/transcripts persisted for debugging
- **`PROD_MODE`**: Raw data is ephemeral; only structured extractions retained

### Plugin Architecture
Strictly typed dependency injection allows swapping components:
```python
# Inject your own transcriber
platform.set_transcriber(DeepgramTranscriber(api_key="..."))

# Inject your own risk model
platform.set_risk_model(MyCustomBayesianNetwork())

# Add custom agents
platform.register_agent(MyResearchAgent())
```

---

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage: Live Session Mode

```python
import asyncio
from datetime import datetime
from pacing import (
    PacingPlatform,
    OperatingMode,
    SessionMetadata,
    MockAudioProvider,
    UncertaintyAuditor
)

# Initialize platform in DEV mode
platform = PacingPlatform(operating_mode=OperatingMode.DEV_MODE)

# Register agents
auditor = UncertaintyAuditor(confidence_threshold=0.70)
platform.register_agent(auditor)

# Create session metadata
session = SessionMetadata(
    session_id="session_001",
    patient_id="patient_123",
    clinician_id="clinician_456",
    start_time=datetime.now(),
    session_type="counseling"
)

# Start live session with mock audio
audio = MockAudioProvider(total_duration_sec=30.0)

async def run_session():
    await platform.start_live_session(session, audio)
    # Session runs until audio completes or stopped manually
    await asyncio.sleep(35)
    await platform.stop_live_session()

    # Check what was flagged
    review_queue = auditor.get_review_queue()
    print(f"Flagged {len(review_queue)} segments for review:")
    for item in review_queue:
        print(f"  - [{item.priority}] {item.transcription.text}")
        print(f"    Reason: {item.reason}")

asyncio.run(run_session())
```

### Basic Usage: Risk Assessment

```python
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
    InterventionType
)

# Initialize platform
platform = PacingPlatform()

# Create patient graph
patient = PatientGraph(
    patient_id="patient_123",
    events=[
        Event(
            event_id="evt_001",
            event_type=EventType.JOB_CHANGE,
            description="Lost job due to budget cuts",
            date=datetime.now() - timedelta(days=15),
            impact_score=-0.7
        )
    ],
    substance_use_records=[
        SubstanceUse(
            use_id="use_001",
            substance_type=SubstanceType.OPIOIDS,
            status=SubstanceUseStatus.REMISSION,
            date=datetime.now() - timedelta(days=180),
            notes="6 months sober"
        )
    ],
    interventions=[
        Intervention(
            intervention_id="int_001",
            intervention_type=InterventionType.MEDICATION,
            description="Buprenorphine 8mg daily",
            start_date=datetime.now() - timedelta(days=180),
            effectiveness_score=0.85
        )
    ]
)

# Calculate risk
report = platform.calculate_risk(patient)

print(f"Risk Score: {report.risk_score:.2%}")
print("\nContributing Factors:")
for factor in report.risk_factors:
    direction = "↑" if factor.contribution > 0 else "↓"
    print(f"  {direction} {factor.factor_name}: {abs(factor.contribution):.2%}")
    for evidence in factor.evidence:
        print(f"      - {evidence}")
```

### Basic Usage: What-If Simulation

```python
from pacing import (
    PacingPlatform,
    create_stable_housing_mutation,
    create_employment_mutation,
    create_mat_intervention_mutation
)

# Initialize platform with simulation-capable model
platform = PacingPlatform()

# Get simulation context
sim = platform.get_simulation_context(patient)

# Scenario 1: What if housing stabilizes?
result = sim.simulate_mutation(create_stable_housing_mutation())

print(f"Current Risk: {result['baseline_risk']:.2%}")
print(f"With Stable Housing: {result['modified_risk']:.2%}")
print(f"Risk Change: {result['delta']:.2%}")
print(f"\n{result['explanation']}")

# Scenario 2: Compare multiple scenarios
scenarios = {
    "Baseline": [],
    "Stable Housing": [create_stable_housing_mutation()],
    "Housing + Employment": [
        create_stable_housing_mutation(),
        create_employment_mutation(employed=True)
    ],
    "Full Support": [
        create_stable_housing_mutation(),
        create_employment_mutation(employed=True),
        create_mat_intervention_mutation()
    ]
}

comparison = sim.compare_scenarios(scenarios)

print("\n📊 Scenario Comparison:")
for scenario_name in comparison["ranked"]:
    result = comparison["scenarios"][scenario_name]
    print(f"  {scenario_name}: {result['modified_risk']:.2%}")

print(f"\n✅ Best Scenario: {comparison['best_scenario']}")
```

---

## 🔌 Extending PACING

### Adding a Custom Transcriber

```python
from pacing.core.transcription_interfaces import ITranscriber
from pacing.models.data_models import TranscriptionResult
import numpy as np

class DeepgramTranscriber(ITranscriber):
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize Deepgram client...

    async def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        is_final: bool = False
    ) -> TranscriptionResult:
        # Call Deepgram API...
        # Return TranscriptionResult with text and confidence
        pass

    def supports_speaker_diarization(self) -> bool:
        return True

# Use it
platform.set_transcriber(DeepgramTranscriber(api_key="your_key"))
```

### Adding a Custom Risk Model

```python
from pacing.core.model_interfaces import ISimulationModel
from pacing.models.data_models import PatientGraph, RiskReport

class MyBayesianNetwork(ISimulationModel):
    def calculate_risk(
        self,
        patient_data: PatientGraph,
        options = None
    ) -> RiskReport:
        # Your probabilistic inference logic
        # (e.g., using pgmpy, PyMC, or custom implementation)
        pass

    def calculate_risk_delta(self, baseline, modified, options=None):
        # Compare baseline vs modified scenarios
        pass

# Use it
platform.set_risk_model(MyBayesianNetwork())
```

### Adding a Custom Sidecar Agent

```python
from pacing.core.agent_interfaces import ISidecarAgent
from pacing.models.data_models import TranscriptionResult

class GuideAgent(ISidecarAgent):
    """Monitors for missing critical information."""

    def __init__(self):
        self.required_topics = {"housing", "employment", "medications"}
        self.covered_topics = set()

    async def on_transcription_update(
        self,
        transcription: TranscriptionResult,
        context = None
    ):
        text_lower = transcription.text.lower()

        # Check if any required topic is mentioned
        for topic in self.required_topics:
            if topic in text_lower:
                self.covered_topics.add(topic)

        # Alert if topics are missing
        missing = self.required_topics - self.covered_topics
        if missing:
            print(f"[Guide] Consider asking about: {', '.join(missing)}")

    def get_agent_name(self) -> str:
        return "GuideAgent"

# Use it
platform.register_agent(GuideAgent())
```

---

## 📁 Project Structure

```
pacing/
├── core/                         # Abstract interfaces (ABCs)
│   ├── audio_interfaces.py       # IAudioProvider
│   ├── transcription_interfaces.py  # ITranscriber
│   ├── agent_interfaces.py       # ISidecarAgent, IScribeAgent
│   ├── model_interfaces.py       # IRiskModel, ISimulationModel
│   └── session_interfaces.py     # ISessionStream
│
├── models/                       # Pydantic data models
│   └── data_models.py            # TranscriptionResult, PatientGraph, RiskReport, etc.
│
├── impl/                         # Concrete implementations
│   ├── defaults/                 # Mock implementations
│   │   ├── mock_audio.py         # MockAudioProvider, ScriptedAudioProvider
│   │   ├── mock_transcriber.py   # MockTranscriber, AdaptiveConfidenceTranscriber
│   │   └── mock_risk_model.py    # MockBayesianModel, MockSimulationModel
│   │
│   └── agents/                   # Sidecar agents
│       └── uncertainty_auditor.py  # UncertaintyAuditor (flags low-confidence)
│
├── privacy/                      # Data handling & redaction
│   └── (future: retention policies, PHI filtering)
│
├── simulation/                   # What-If analysis engine
│   └── simulation_engine.py      # SimulationContext, Mutation
│
└── platform.py                   # PacingPlatform (main orchestrator)
```

---

## 🔒 Privacy & Security

### Operating Modes

| Mode | Raw Audio | Transcripts | Extracted Entities |
|------|-----------|-------------|-------------------|
| **DEV_MODE** | Persisted | Persisted | Persisted |
| **PROD_MODE** | Ephemeral | Ephemeral | Persisted |

### Recommendations for Production

1. **Use PROD_MODE** unless debugging
2. **Implement PHI filtering** in the privacy module
3. **Encrypt stored entities** at rest
4. **Set retention policies** for review queues
5. **Audit access** to patient graphs
6. **Obtain informed consent** for AI assistance

---

## 🧪 Testing

### Running Mock Demonstrations

The platform includes mock implementations for testing without external dependencies:

```python
# Mock audio (generates synthetic audio)
audio = MockAudioProvider(total_duration_sec=60.0)

# Mock transcriber (returns scripted text)
transcriber = MockTranscriber(latency_ms=50)

# Mock risk model (rule-based heuristics)
model = MockBayesianModel(base_risk=0.50)
```

### Testing the Auditor

```python
from pacing import MockTranscriber, AdaptiveConfidenceTranscriber

# Use adaptive transcriber to generate low-confidence segments
transcriber = AdaptiveConfidenceTranscriber()

# The auditor will automatically flag segments with:
# - Confidence < 0.70
# - Medical terms (even if confidence is acceptable)
```

---

## 🎓 Research Use Cases

PACING is designed to support clinical research:

### Example 1: Studying Information Gathering Patterns
```python
# Add a custom agent to track which topics are discussed
class TopicTrackingAgent(ISidecarAgent):
    def __init__(self):
        self.topic_mentions = {}

    async def on_transcription_update(self, transcription, context):
        # Count topic mentions over time
        pass
```

### Example 2: Testing Risk Model Variants
```python
# Compare multiple risk models on the same patient data
models = {
    "Rule-Based": RuleBasedModel(),
    "Bayesian Network": BayesianNetworkModel(),
    "Random Forest": MLModel()
}

for name, model in models.items():
    platform.set_risk_model(model)
    report = platform.calculate_risk(patient_graph)
    print(f"{name}: {report.risk_score:.2%}")
```

### Example 3: Evaluating Simulation Accuracy
```python
# Compare predicted vs actual outcomes
sim = platform.get_simulation_context(patient_baseline)
prediction = sim.simulate_mutation(housing_mutation)

# Later, compare to actual outcome
actual_outcome = load_patient_data_at_followup(patient_id, months=6)
actual_report = platform.calculate_risk(actual_outcome)

accuracy = abs(prediction['modified_risk'] - actual_report.risk_score)
print(f"Prediction error: {accuracy:.2%}")
```

---

## 🛠️ Development Roadmap

### Planned Features
- [ ] Guide Agent implementation
- [ ] Scribe Agent with LLM-based entity extraction
- [ ] PHI filtering in privacy module
- [ ] Session persistence & replay
- [ ] Web dashboard for review queues
- [ ] Integration with EHR systems (FHIR)
- [ ] Real audio providers (PyAudio, sounddevice)
- [ ] Production transcribers (Deepgram, Whisper)
- [ ] Real probabilistic models (pgmpy, PyMC)

### Contributing
This is a research scaffold. To add features:
1. Define interfaces in `pacing/core/`
2. Implement in `pacing/impl/`
3. Add tests demonstrating usage
4. Update this README with examples

---

## 📚 Technical Background

### Why "Glass Box" AI?
In clinical settings, unexplainable AI creates liability and trust issues. PACING enforces:
- **Explainability**: Models must list contributing factors
- **Auditability**: Human-in-the-loop verification of uncertain data
- **Traceability**: All decisions have evidence trails

### Why Separate "Raw" from "Extracted" Data?
Audio/transcripts are high-dimensional and privacy-sensitive. Structured extractions (medications, events) are:
- Lower-dimensional (easier to analyze)
- Easier to anonymize
- Sufficient for most downstream tasks

This separation enables PROD_MODE's ephemeral raw data policy.

### Why Plugin Architecture?
Clinical research requires flexibility:
- Test multiple transcription APIs
- Compare different risk models
- Add domain-specific agents
- Adapt to new research questions

The interface-based design makes these variations trivial.

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🤝 Citation

If you use PACING in research, please cite:

```bibtex
@software{pacing2026,
  title={PACING: Platform for Agentic Clinical Intelligence with Networked Graphs},
  author={Thor Whalen},
  year={2026},
  url={https://github.com/thorwhalen/pacing}
}
```

---

**⚕️ Clinical Disclaimer**: PACING is a research tool. It is not FDA-approved and should not be used as the sole basis for clinical decisions. Always defer to trained clinicians and established clinical protocols.
