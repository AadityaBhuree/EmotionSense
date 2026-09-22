"""Multimodal Clinical Reasoning Agent with Chain-of-Thought Diagnostic Synthesis."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.agent.llm_provider import BaseLLMProvider, LLMProviderFactory, RuleBasedExpertProvider
from src.agent.models import (
    AffectiveObservation,
    ClinicalRiskLevel,
    ClinicalRiskScore,
    DiagnosticSynthesis,
    InterventionRecommendation,
    InterventionUrgency,
    LLMProviderConfig,
    TurningPoint,
)

logger = logging.getLogger("EmotionSense.Agent.ClinicalAgent")


class ClinicalReasoningAgent:
    """Enterprise-grade agent synthesizing multimodal affective telemetry into clinical insights."""

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        provider_config: Optional[LLMProviderConfig] = None,
    ):
        self.provider = provider or LLMProviderFactory.get_provider(provider_config)
        self.fallback = RuleBasedExpertProvider()

    def synthesize_session(self, session_data: Dict[str, Any]) -> DiagnosticSynthesis:
        """Analyze complete session telemetry and produce a validated DiagnosticSynthesis."""
        session_id = session_data.get("session_id", "sess_unknown")
        subject_id = session_data.get("candidate_id") or session_data.get("subject_id") or "Anonymous"
        assessment_type = session_data.get("assessment_type", "Clinical Evaluation")

        # 1. Compress and extract high-signal telemetry
        samples = session_data.get("timeline_samples", [])
        anomalies = session_data.get("anomalies", [])
        turning_points = self._detect_turning_points(samples)
        telemetry_summary = self._summarize_telemetry(samples, anomalies, turning_points)

        # 2. Formulate Chain-of-Thought Prompt
        system_prompt = (
            "You are a Board-Certified Clinical Neuropsychologist and Affective Computing Specialist. "
            "Your objective is to evaluate objective multimodal affective telemetry (Valence-Arousal-Dominance, "
            "Facial Action Units, Acoustic Pitch/Jitter Prosody, and Affective Anomalies) to synthesize "
            "an actionable, evidence-based diagnostic assessment. Output ONLY valid JSON."
        )

        user_prompt = f"""
Session ID: {session_id}
Subject ID: {subject_id}
Assessment Type: {assessment_type}

### Telemetry Summary:
{json.dumps(telemetry_summary, indent=2)}

### Critical Anomalies Detected:
{json.dumps(anomalies[:5], indent=2) if anomalies else "None"}

Please evaluate this session and provide a structured JSON response matching the following schema:
{{
  "executive_summary": "Comprehensive 2-3 sentence clinical overview",
  "risk_assessment": {{
    "risk_level": "MINIMAL" | "MILD_DISTRESS" | "MODERATE_RISK" | "ACUTE_CRISIS" | "COGNITIVE_OVERLOAD",
    "overall_score": float (0-100),
    "distress_index": float (0.0-1.0),
    "fatigue_index": float (0.0-1.0),
    "volatility_index": float (0.0-1.0),
    "primary_concerns": ["concern 1", "concern 2"]
  }},
  "observations": [
    {{
      "domain": "facial" | "acoustic" | "semantic" | "dyadic",
      "finding": "objective description",
      "clinical_significance": "interpretation",
      "confidence": float (0.0-1.0)
    }}
  ],
  "interventions": [
    {{
      "action": "recommended intervention",
      "urgency": "ROUTINE" | "RECOMMENDED" | "URGENT" | "IMMEDIATE",
      "rationale": "clinical reasoning",
      "contraindications": "optional contraindications or null"
    }}
  ],
  "prognosis": "Clinical trajectory prognosis"
}}
"""

        # 3. Call LLM provider
        raw_output = self.provider.generate(user_prompt, system_prompt)

        # 4. Parse JSON and validate
        synthesis = self._parse_and_validate(raw_output, session_id, subject_id, assessment_type, turning_points)
        return synthesis

    def triage_live_state(
        self,
        current_sample: Dict[str, Any],
        recent_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Rapid in-session affective triage assessing acute distress or cognitive fatigue."""
        v = current_sample.get("valence", 0.0)
        a = current_sample.get("arousal", 0.0)
        d = current_sample.get("dominance", 0.0)
        emotion = current_sample.get("dominant_emotion", "neutral")

        # Compute acute distress trigger
        is_acute_distress = (v < -0.45 and a > 0.40) or emotion in ("fear", "anger")
        is_fatigue = (a < -0.40 and d < -0.30)

        triage_status = "STABLE"
        alert_message = "Affective baseline within normal limits."
        action_hint = "Proceed with standard protocol."

        if is_acute_distress:
            triage_status = "ELEVATED_DISTRESS"
            alert_message = f"Acute distress detected: Valence ({v:.2f}) with autonomic arousal spike ({a:.2f})."
            action_hint = "Consider pausing, validating emotions, or switching to an open-ended grounding topic."
        elif is_fatigue:
            triage_status = "COGNITIVE_FATIGUE"
            alert_message = f"Cognitive fatigue pattern: Low arousal ({a:.2f}) and diminished engagement."
            action_hint = "Consider offering a brief rest break or active re-engagement check."

        return {
            "triage_status": triage_status,
            "alert_message": alert_message,
            "action_hint": action_hint,
            "current_metrics": {"valence": v, "arousal": a, "dominance": d, "emotion": emotion},
            "timestamp": current_sample.get("timestamp_sec", 0.0),
        }

    def _detect_turning_points(self, samples: List[Dict[str, Any]]) -> List[TurningPoint]:
        """Detect sharp velocity shifts in the emotional trajectory."""
        if len(samples) < 3:
            return []

        points = []
        for i in range(1, len(samples)):
            prev_s = samples[i - 1]
            curr_s = samples[i]

            prev_v = prev_s.get("valence", 0.0)
            curr_v = curr_s.get("valence", 0.0)
            prev_a = prev_s.get("arousal", 0.0)
            curr_a = curr_s.get("arousal", 0.0)

            dv = curr_v - prev_v
            da = curr_a - prev_a
            ts = curr_s.get("timestamp_sec", float(i))

            if dv <= -0.40:
                points.append(
                    TurningPoint(
                        timestamp_sec=ts,
                        event="Acute Valence Drop",
                        description=f"Sharp negative decline (ΔValence: {dv:+.2f}). Emotional dip detected.",
                        valence_delta=round(dv, 2),
                        arousal_delta=round(da, 2),
                    )
                )
            elif da >= 0.50:
                points.append(
                    TurningPoint(
                        timestamp_sec=ts,
                        event="Arousal Surge",
                        description=f"Sudden sympathetic activation (ΔArousal: {da:+.2f}). Stress or surprise flare.",
                        valence_delta=round(dv, 2),
                        arousal_delta=round(da, 2),
                    )
                )

        return points[:5]

    def _summarize_telemetry(
        self,
        samples: List[Dict[str, Any]],
        anomalies: List[Dict[str, Any]],
        turning_points: List[TurningPoint],
    ) -> Dict[str, Any]:
        """Compress timeline samples into key clinical statistical landmarks."""
        if not samples:
            return {
                "sample_count": 0,
                "mean_valence": 0.0,
                "mean_arousal": 0.0,
                "valence_range": [-0.1, 0.1],
                "arousal_range": [-0.1, 0.1],
                "anomalies_count": len(anomalies),
                "turning_points_count": len(turning_points),
            }

        valences = [s.get("valence", 0.0) for s in samples]
        arousals = [s.get("arousal", 0.0) for s in samples]

        return {
            "sample_count": len(samples),
            "mean_valence": round(sum(valences) / len(valences), 3),
            "mean_arousal": round(sum(arousals) / len(arousals), 3),
            "min_valence": round(min(valences), 3),
            "max_valence": round(max(valences), 3),
            "valence_volatility": round(
                (sum((x - sum(valences) / len(valences)) ** 2 for x in valences) / len(valences)) ** 0.5, 3
            ),
            "anomalies_count": len(anomalies),
            "turning_points_count": len(turning_points),
        }

    def _parse_and_validate(
        self,
        raw_output: str,
        session_id: str,
        subject_id: str,
        assessment_type: str,
        turning_points: List[TurningPoint],
    ) -> DiagnosticSynthesis:
        """Parse raw model output into structured DiagnosticSynthesis with graceful fallback."""
        try:
            # Strip markdown fences if present
            cleaned = raw_output.strip()
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1)

            data = json.loads(cleaned)

            # Build models
            risk_data = data.get("risk_assessment", {})
            risk_assessment = ClinicalRiskScore(
                risk_level=ClinicalRiskLevel(risk_data.get("risk_level", "MINIMAL")),
                overall_score=float(risk_data.get("overall_score", 20.0)),
                distress_index=float(risk_data.get("distress_index", 0.1)),
                fatigue_index=float(risk_data.get("fatigue_index", 0.1)),
                volatility_index=float(risk_data.get("volatility_index", 0.1)),
                primary_concerns=risk_data.get("primary_concerns", []),
            )

            observations = [
                AffectiveObservation(
                    domain=o.get("domain", "systemic"),
                    finding=o.get("finding", ""),
                    clinical_significance=o.get("clinical_significance", ""),
                    confidence=float(o.get("confidence", 0.9)),
                )
                for o in data.get("observations", [])
            ]

            interventions = [
                InterventionRecommendation(
                    action=item.get("action", ""),
                    urgency=InterventionUrgency(item.get("urgency", "ROUTINE")),
                    rationale=item.get("rationale", ""),
                    contraindications=item.get("contraindications"),
                )
                for item in data.get("interventions", [])
            ]

            return DiagnosticSynthesis(
                session_id=session_id,
                subject_id=subject_id,
                assessment_type=assessment_type,
                executive_summary=data.get("executive_summary", "Session completed."),
                risk_assessment=risk_assessment,
                observations=observations,
                interventions=interventions,
                turning_points=turning_points,
                prognosis=data.get("prognosis", "Stable baseline."),
                provider_used=self.provider.provider_name,
                model_name=self.provider.model_name,
            )
        except Exception as e:
            logger.warning("Failed parsing agent response (%s); utilizing expert deterministic synthesis", e)
            fallback_json = json.loads(self.fallback.generate(f"session_id: {session_id}"))
            fallback_risk = ClinicalRiskScore(**fallback_json["risk_assessment"])
            return DiagnosticSynthesis(
                session_id=session_id,
                subject_id=subject_id,
                assessment_type=assessment_type,
                executive_summary=fallback_json["executive_summary"],
                risk_assessment=fallback_risk,
                observations=[AffectiveObservation(**o) for o in fallback_json["observations"]],
                interventions=[
                    InterventionRecommendation(
                        action=i["action"],
                        urgency=InterventionUrgency(i["urgency"]),
                        rationale=i["rationale"],
                        contraindications=i.get("contraindications"),
                    )
                    for i in fallback_json["interventions"]
                ],
                turning_points=turning_points,
                prognosis=fallback_json["prognosis"],
                provider_used="RuleBasedExpert",
                model_name="expert-deterministic-v1",
            )
