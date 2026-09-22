"""Unit and integration tests for Phase 9: Multimodal Agentic Reasoning & Clinical Copilot."""

from typing import Any, Dict

from src.agent.models import (
    ClinicalRiskLevel,
    InterventionUrgency,
    ClinicalRiskScore,
    InterventionRecommendation,
    AffectiveObservation,
    TurningPoint,
    DiagnosticSynthesis,
    ChatCopilotQuery,
    LLMProviderConfig,
)
from src.agent.llm_provider import (
    BaseLLMProvider,
    RuleBasedExpertProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    GeminiProvider,
    LLMProviderFactory,
)
from src.agent.clinical_agent import ClinicalReasoningAgent
from src.agent.chat_copilot import ClinicalChatCopilot


def test_models_instantiation_and_validation():
    """Verify data model construction, constraints, and serialization."""
    risk = ClinicalRiskScore(
        risk_level=ClinicalRiskLevel.MODERATE_RISK,
        overall_score=65.0,
        distress_index=0.55,
        fatigue_index=0.20,
        volatility_index=0.40,
        primary_concerns=["Transient distress spike"],
    )
    assert risk.risk_level == ClinicalRiskLevel.MODERATE_RISK
    assert risk.overall_score == 65.0

    obs = AffectiveObservation(
        domain="facial",
        finding="Corrugator tension detected",
        clinical_significance="Mild cognitive load",
        confidence=0.88,
    )
    assert obs.confidence == 0.88

    rec = InterventionRecommendation(
        action="Provide grounding exercise",
        urgency=InterventionUrgency.RECOMMENDED,
        rationale="Elevated arousal spike",
    )
    assert rec.urgency == InterventionUrgency.RECOMMENDED

    synth = DiagnosticSynthesis(
        session_id="test_001",
        executive_summary="Participant maintained stable affect with minor stress spikes.",
        risk_assessment=risk,
        observations=[obs],
        interventions=[rec],
        turning_points=[TurningPoint(timestamp_sec=12.5, event="Drop", description="Dip in valence", valence_delta=-0.45)],
    )
    assert synth.session_id == "test_001"
    assert len(synth.observations) == 1
    assert len(synth.turning_points) == 1


def test_rule_based_provider_generation():
    """Test deterministic rule-based LLM provider produces valid JSON with expected structure."""
    provider = RuleBasedExpertProvider()
    
    # Low distress prompt
    low_res = provider.generate("Evaluate session 101 with calm affect")
    assert "executive_summary" in low_res
    assert "risk_assessment" in low_res
    assert "MINIMAL" in low_res

    # High distress prompt
    high_res = provider.generate("Session indicates severe valence_crash and acute distress")
    assert "MODERATE_RISK" in high_res
    assert "cognitive defusion" in high_res


def test_provider_factory():
    """Test factory resolution of configured providers."""
    rule_p = LLMProviderFactory.get_provider(LLMProviderConfig(provider_name="rule_based"))
    assert isinstance(rule_p, RuleBasedExpertProvider)

    ollama_p = LLMProviderFactory.get_provider(LLMProviderConfig(provider_name="ollama"))
    assert isinstance(ollama_p, OllamaProvider)

    openai_p = LLMProviderFactory.get_provider(LLMProviderConfig(provider_name="openai"))
    assert isinstance(openai_p, OpenAICompatibleProvider)

    gemini_p = LLMProviderFactory.get_provider(LLMProviderConfig(provider_name="gemini"))
    assert isinstance(gemini_p, GeminiProvider)


def test_clinical_agent_synthesis():
    """Test full session diagnostic synthesis using the ClinicalReasoningAgent."""
    agent = ClinicalReasoningAgent()

    mock_session: Dict[str, Any] = {
        "session_id": "sess_unit_test",
        "candidate_id": "Cand_42",
        "assessment_type": "Technical Interview",
        "timeline_samples": [
            {"timestamp_sec": 1.0, "valence": 0.3, "arousal": 0.1, "dominance": 0.2, "dominant_emotion": "neutral"},
            {"timestamp_sec": 2.0, "valence": 0.4, "arousal": 0.2, "dominance": 0.3, "dominant_emotion": "joy"},
            {"timestamp_sec": 3.0, "valence": -0.2, "arousal": 0.6, "dominance": -0.1, "dominant_emotion": "fear"},
            {"timestamp_sec": 4.0, "valence": 0.2, "arousal": 0.1, "dominance": 0.1, "dominant_emotion": "neutral"},
        ],
        "anomalies": [
            {"timestamp_sec": 3.0, "anomaly_type": "valence_crash", "valence": -0.2, "arousal": 0.6}
        ],
    }

    synthesis = agent.synthesize_session(mock_session)
    assert synthesis.session_id == "sess_unit_test"
    assert synthesis.subject_id == "Cand_42"
    assert synthesis.assessment_type == "Technical Interview"
    assert synthesis.risk_assessment.overall_score >= 0.0
    assert len(synthesis.observations) >= 1
    assert len(synthesis.interventions) >= 1
    assert len(synthesis.turning_points) >= 1
    assert synthesis.turning_points[0].event == "Acute Valence Drop"


def test_clinical_agent_live_triage():
    """Verify live stream telemetry triage for immediate alerts."""
    agent = ClinicalReasoningAgent()

    # 1. Normal state
    norm_triage = agent.triage_live_state(
        {"valence": 0.4, "arousal": 0.2, "dominance": 0.3, "dominant_emotion": "joy"},
        []
    )
    assert norm_triage["triage_status"] == "STABLE"

    # 2. Acute distress
    distress_triage = agent.triage_live_state(
        {"valence": -0.6, "arousal": 0.7, "dominance": -0.4, "dominant_emotion": "fear"},
        []
    )
    assert distress_triage["triage_status"] == "ELEVATED_DISTRESS"
    assert "grounding" in distress_triage["action_hint"].lower()

    # 3. Cognitive fatigue
    fatigue_triage = agent.triage_live_state(
        {"valence": -0.1, "arousal": -0.5, "dominance": -0.4, "dominant_emotion": "neutral"},
        []
    )
    assert fatigue_triage["triage_status"] == "COGNITIVE_FATIGUE"


def test_chat_copilot_query():
    """Verify interactive clinical copilot question handling and follow-up generation."""
    copilot = ClinicalChatCopilot()

    ctx = {
        "session_id": "sess_copilot_test",
        "candidate_id": "Alice",
        "timeline_samples": [{"valence": 0.1, "arousal": 0.2}] * 20,
        "anomalies": [{"timestamp_sec": 14.2, "anomaly_type": "hyper_arousal", "valence": 0.1}],
    }

    # Query 1: Why arousal
    q1 = ChatCopilotQuery(session_id="sess_copilot_test", question="Why did arousal spike?")
    res1 = copilot.query(q1, ctx)
    assert "arousal" in res1.answer.lower() or "jitter" in res1.answer.lower()
    assert len(res1.suggested_followups) >= 2

    # Query 2: Risk overview
    q2 = ChatCopilotQuery(session_id="sess_copilot_test", question="What are the clinical risks?")
    res2 = copilot.query(q2, ctx)
    assert "risk" in res2.answer.lower() or "anomalous" in res2.answer.lower()
    assert len(res2.suggested_followups) >= 2


def test_corrupted_output_fallback():
    """Ensure agent gracefully falls back to deterministic rule engine when provider returns garbage."""
    class BrokenProvider(BaseLLMProvider):
        def generate(self, prompt: str, system_prompt: str = None) -> str:
            return "This is not JSON at all! Just raw corrupted text."

    agent = ClinicalReasoningAgent(provider=BrokenProvider())
    synth = agent.synthesize_session({"session_id": "sess_broken"})

    assert synth.session_id == "sess_broken"
    assert synth.provider_used == "RuleBasedExpert"
    assert synth.risk_assessment.overall_score >= 0.0
    assert len(synth.observations) >= 1
