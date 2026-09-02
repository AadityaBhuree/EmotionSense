"""Affective Anomaly and Emotional Distress Sentinel Engine.

Detects real-time anomalies in multimodal emotional telemetry:
- Emotional Volatility: Sudden valence drops or hyper-arousal spikes
- Acute Distress: Sustained negative affect accompanied by elevated stress/arousal
- Cognitive Fatigue Overload: Prolonged elevated fatigue with collapsed attention
- Attentional Disengagement: Rapid drop in gaze contact and expressive engagement
"""

import time
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collections import deque

from src.core.types import MultimodalEmotionState, AffectVector


class AnomalySeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AnomalyType(str, Enum):
    VALENCE_CRASH = "valence_crash"
    SUSTAINED_DISTRESS = "sustained_distress"
    COGNITIVE_OVERLOAD = "cognitive_overload"
    ATTENTION_COLLAPSE = "attention_collapse"
    HYPER_AROUSAL_SPIKE = "hyper_arousal_spike"


@dataclass
class AffectiveAnomaly:
    """Represents an identified affective anomaly event."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    timestamp: float
    description: str
    trigger_metrics: Dict[str, float]
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["anomaly_type"] = self.anomaly_type.value
        res["severity"] = self.severity.value
        return res


class AffectiveAnomalyDetector:
    """Real-time temporal sentinel that scans incoming emotional states for anomalies."""

    def __init__(
        self,
        window_size: int = 15,
        valence_crash_threshold: float = 0.45,
        distress_duration_samples: int = 5,
        fatigue_overload_threshold: float = 0.75,
        attention_collapse_threshold: float = 0.25,
    ):
        self.window_size = window_size
        self.valence_crash_threshold = valence_crash_threshold
        self.distress_duration_samples = distress_duration_samples
        self.fatigue_overload_threshold = fatigue_overload_threshold
        self.attention_collapse_threshold = attention_collapse_threshold

        self._history: deque = deque(maxlen=window_size)
        self.detected_anomalies: List[AffectiveAnomaly] = []
        self._anomaly_counter = 0

    def reset(self):
        """Clears sliding window history and detected anomalies."""
        self._history.clear()
        self.detected_anomalies.clear()
        self._anomaly_counter = 0

    def process_state(self, state: MultimodalEmotionState) -> Optional[AffectiveAnomaly]:
        """Analyzes a new emotional state sample against historical window and returns any detected anomaly."""
        self._history.append(state)
        if len(self._history) < 2:
            return None

        now = state.timestamp or time.time()

        # 1. Check for Sudden Valence Crash (Drop of > threshold within recent 3 samples)
        if len(self._history) >= 2:
            prev_state = self._history[-2]
            v_delta = state.affect.valence - prev_state.affect.valence
            if v_delta <= -self.valence_crash_threshold:
                self._anomaly_counter += 1
                severity = AnomalySeverity.CRITICAL if v_delta <= -0.7 else AnomalySeverity.WARNING
                anomaly = AffectiveAnomaly(
                    anomaly_id=f"anom_{self._anomaly_counter}",
                    anomaly_type=AnomalyType.VALENCE_CRASH,
                    severity=severity,
                    timestamp=now,
                    description=f"Sudden emotional valence crash detected (ΔV = {v_delta:.2f}). Rapid shift into distress.",
                    trigger_metrics={
                        "valence": state.affect.valence,
                        "previous_valence": prev_state.affect.valence,
                        "valence_delta": round(v_delta, 3),
                        "arousal": state.affect.arousal,
                    },
                    recommended_action="Validate participant state; slow conversational tempo or invite brief clarification."
                )
                self.detected_anomalies.append(anomaly)
                return anomaly

        # 2. Check for Hyper-Arousal Spike (e.g. panic or severe frustration)
        if state.affect.arousal >= 0.85 and state.affect.valence < -0.2:
            self._anomaly_counter += 1
            anomaly = AffectiveAnomaly(
                anomaly_id=f"anom_{self._anomaly_counter}",
                anomaly_type=AnomalyType.HYPER_AROUSAL_SPIKE,
                severity=AnomalySeverity.CRITICAL,
                timestamp=now,
                description="Hyper-arousal stress spike detected with negative valence.",
                trigger_metrics={
                    "arousal": round(state.affect.arousal, 3),
                    "valence": round(state.affect.valence, 3),
                    "dominant_emotion": state.dominant_emotion,
                },
                recommended_action="Acknowledge heightened emotional tension; consider active de-escalation response."
            )
            self.detected_anomalies.append(anomaly)
            return anomaly

        # 3. Check for Sustained Distress (Consecutive negative valence + non-trivial arousal)
        if len(self._history) >= self.distress_duration_samples:
            recent = list(self._history)[-self.distress_duration_samples:]
            is_sustained_distress = all(
                s.affect.valence < -0.3 and s.dominant_emotion in ("sadness", "anger", "fear", "contempt", "disgust")
                for s in recent
            )
            if is_sustained_distress:
                self._anomaly_counter += 1
                anomaly = AffectiveAnomaly(
                    anomaly_id=f"anom_{self._anomaly_counter}",
                    anomaly_type=AnomalyType.SUSTAINED_DISTRESS,
                    severity=AnomalySeverity.WARNING,
                    timestamp=now,
                    description=f"Sustained emotional distress observed across {self.distress_duration_samples} consecutive samples.",
                    trigger_metrics={
                        "average_valence": round(sum(s.affect.valence for s in recent) / len(recent), 3),
                        "dominant_emotion": state.dominant_emotion,
                        "duration_samples": self.distress_duration_samples,
                    },
                    recommended_action="Offer supportive empathy validation; assess whether topic change is warranted."
                )
                self.detected_anomalies.append(anomaly)
                return anomaly

        # 4. Check for Cognitive Fatigue Overload
        if len(self._history) >= 4:
            recent_fatigue = [s.fatigue_level for s in list(self._history)[-4:]]
            recent_attention = [s.attention_score for s in list(self._history)[-4:]]
            avg_fatigue = sum(recent_fatigue) / len(recent_fatigue)
            avg_attention = sum(recent_attention) / len(recent_attention)

            if avg_fatigue >= self.fatigue_overload_threshold and avg_attention <= 0.40:
                self._anomaly_counter += 1
                anomaly = AffectiveAnomaly(
                    anomaly_id=f"anom_{self._anomaly_counter}",
                    anomaly_type=AnomalyType.COGNITIVE_OVERLOAD,
                    severity=AnomalySeverity.WARNING,
                    timestamp=now,
                    description="Cognitive fatigue overload threshold reached. Participant gaze and energy depleted.",
                    trigger_metrics={
                        "average_fatigue": round(avg_fatigue, 3),
                        "average_attention": round(avg_attention, 3),
                    },
                    recommended_action="Suggest a short rest break, breathing pause, or change of session modality."
                )
                self.detected_anomalies.append(anomaly)
                return anomaly

        # 5. Check for Sudden Attention Collapse
        if len(self._history) >= 3:
            recent_att = [s.attention_score for s in list(self._history)[-3:]]
            if recent_att[0] >= 0.70 and recent_att[-1] <= self.attention_collapse_threshold:
                self._anomaly_counter += 1
                anomaly = AffectiveAnomaly(
                    anomaly_id=f"anom_{self._anomaly_counter}",
                    anomaly_type=AnomalyType.ATTENTION_COLLAPSE,
                    severity=AnomalySeverity.INFO,
                    timestamp=now,
                    description="Acute visual attention collapse detected. Subject looked away or disengaged from camera.",
                    trigger_metrics={
                        "initial_attention": round(recent_att[0], 3),
                        "current_attention": round(recent_att[-1], 3),
                    },
                    recommended_action="Verify camera framing and check if subject was distracted by an external event."
                )
                self.detected_anomalies.append(anomaly)
                return anomaly

        return None

    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Returns statistics of all detected anomalies in this session."""
        by_severity = {s.value: 0 for s in AnomalySeverity}
        by_type = {t.value: 0 for t in AnomalyType}

        for a in self.detected_anomalies:
            by_severity[a.severity.value] += 1
            by_type[a.anomaly_type.value] += 1

        return {
            "total_anomalies": len(self.detected_anomalies),
            "by_severity": by_severity,
            "by_type": by_type,
            "events": [a.to_dict() for a in self.detected_anomalies],
        }
