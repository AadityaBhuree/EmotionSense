"""Unit tests for DiagnosticReportGenerator."""

import pytest
import time
from src.core.types import SessionRecord
from src.utils.report_generator import DiagnosticReportGenerator


@pytest.fixture
def sample_session_record():
    return SessionRecord(
        session_id="test_session_101",
        start_time=time.time() - 120,
        end_time=time.time(),
        samples_count=50,
        average_affect={"valence": 0.45, "arousal": 0.32, "dominance": 0.50},
        dominant_emotion_distribution={"joy": 0.60, "neutral": 0.30, "surprise": 0.10},
        average_engagement=0.82,
        average_fatigue=0.22,
        average_attention=0.91,
        key_moments=[
            {
                "relative_time_sec": 30.5,
                "dominant_emotion": "joy",
                "confidence": 0.94,
                "valence": 0.72,
                "arousal": 0.55,
            }
        ]
    )


@pytest.fixture
def sample_anomalies():
    return [
        {
            "anomaly_id": "anom_1",
            "anomaly_type": "valence_crash",
            "severity": "CRITICAL",
            "timestamp": time.time() - 60,
            "description": "Sudden emotional valence crash detected.",
            "recommended_action": "Pause conversation and assess tone.",
        }
    ]


def test_markdown_report_generation(sample_session_record, sample_anomalies):
    md = DiagnosticReportGenerator.generate_markdown_report(
        sample_session_record, sample_anomalies
    )
    assert "# 🧠 EmotionSense — Session Diagnostic Report" in md
    assert "test_session_101" in md
    assert "**Mean Engagement Index:** 82.0%" in md
    assert "**Joy**" in md
    assert "CRITICAL" in md
    assert "valence_crash" in md
    assert "Key Affective Pivot Moments" in md


def test_html_report_generation(sample_session_record, sample_anomalies):
    html = DiagnosticReportGenerator.generate_html_report(
        sample_session_record, sample_anomalies
    )
    assert "<!DOCTYPE html>" in html
    assert "test_session_101" in html
    assert "EmotionSense" in html
    assert "82.0%" in html
    assert "CRITICAL" in html
    assert "Sudden emotional valence crash detected" in html


def test_html_report_without_anomalies(sample_session_record):
    html = DiagnosticReportGenerator.generate_html_report(sample_session_record, None)
    assert "No affective anomalies or distress spikes detected" in html
