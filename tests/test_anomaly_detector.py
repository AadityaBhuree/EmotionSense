"""Unit tests for AffectiveAnomalyDetector sentinel engine."""

import pytest
import time
from src.core.types import MultimodalEmotionState, AffectVector
from src.fusion.anomaly_detector import (
    AffectiveAnomalyDetector,
    AnomalySeverity,
    AnomalyType,
)


@pytest.fixture
def detector():
    return AffectiveAnomalyDetector(
        window_size=10,
        valence_crash_threshold=0.45,
        distress_duration_samples=3,
        fatigue_overload_threshold=0.75,
        attention_collapse_threshold=0.25,
    )


def test_initialization_and_reset(detector):
    assert len(detector.detected_anomalies) == 0
    state = MultimodalEmotionState(
        dominant_emotion="joy",
        confidence=0.9,
        affect=AffectVector(valence=0.7, arousal=0.5, dominance=0.6),
    )
    anomaly = detector.process_state(state)
    assert anomaly is None
    detector.reset()
    assert len(detector.detected_anomalies) == 0


def test_valence_crash_detection(detector):
    # Baseline positive state
    state1 = MultimodalEmotionState(
        dominant_emotion="joy",
        confidence=0.85,
        affect=AffectVector(valence=0.6, arousal=0.4, dominance=0.5),
    )
    detector.process_state(state1)

    # Abrupt plunge into negative state (ΔV = -0.1 - 0.6 = -0.7)
    state2 = MultimodalEmotionState(
        dominant_emotion="sadness",
        confidence=0.88,
        affect=AffectVector(valence=-0.1, arousal=0.2, dominance=0.1),
    )
    anomaly = detector.process_state(state2)

    assert anomaly is not None
    assert anomaly.anomaly_type == AnomalyType.VALENCE_CRASH
    assert anomaly.severity == AnomalySeverity.CRITICAL
    assert "Sudden emotional valence crash" in anomaly.description


def test_hyper_arousal_spike_detection(detector):
    # State with extreme arousal and negative valence (panic/anger)
    state1 = MultimodalEmotionState(
        dominant_emotion="neutral",
        affect=AffectVector(valence=0.0, arousal=0.1, dominance=0.0),
    )
    detector.process_state(state1)

    state2 = MultimodalEmotionState(
        dominant_emotion="anger",
        confidence=0.95,
        affect=AffectVector(valence=-0.4, arousal=0.9, dominance=0.8),
    )
    anomaly = detector.process_state(state2)

    assert anomaly is not None
    assert anomaly.anomaly_type in (AnomalyType.HYPER_AROUSAL_SPIKE, AnomalyType.VALENCE_CRASH)
    if anomaly.anomaly_type == AnomalyType.HYPER_AROUSAL_SPIKE:
        assert anomaly.severity == AnomalySeverity.CRITICAL


def test_sustained_distress_detection(detector):
    # Feed 3 consecutive distress samples (threshold is set to 3 in fixture)
    for i in range(3):
        state = MultimodalEmotionState(
            dominant_emotion="sadness",
            confidence=0.8,
            affect=AffectVector(valence=-0.5, arousal=-0.1, dominance=-0.3),
        )
        detector.process_state(state)

    summary = detector.get_anomaly_summary()
    assert summary["total_anomalies"] >= 1
    types = [a["anomaly_type"] for a in summary["events"]]
    assert AnomalyType.SUSTAINED_DISTRESS.value in types


def test_cognitive_fatigue_overload(detector):
    # 4 consecutive samples with high fatigue and low attention
    for _ in range(4):
        state = MultimodalEmotionState(
            dominant_emotion="neutral",
            affect=AffectVector(valence=-0.1, arousal=-0.4, dominance=-0.2),
            fatigue_level=0.85,
            attention_score=0.25,
        )
        detector.process_state(state)

    summary = detector.get_anomaly_summary()
    types = [a["anomaly_type"] for a in summary["events"]]
    assert AnomalyType.COGNITIVE_OVERLOAD.value in types


def test_attention_collapse(detector):
    detector.reset()
    # Sample 1: high attention
    detector.process_state(MultimodalEmotionState(attention_score=0.85))
    # Sample 2: moderate attention
    detector.process_state(MultimodalEmotionState(attention_score=0.50))
    # Sample 3: collapsed attention
    anomaly = detector.process_state(MultimodalEmotionState(attention_score=0.15))

    assert anomaly is not None
    assert anomaly.anomaly_type == AnomalyType.ATTENTION_COLLAPSE
    assert anomaly.severity == AnomalySeverity.INFO
