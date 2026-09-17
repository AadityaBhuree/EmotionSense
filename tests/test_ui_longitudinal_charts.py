"""Tests for Phase 7 Longitudinal UI Charts and Component Renderers."""

import pytest
from src.analytics.longitudinal_analyzer import (
    LongitudinalProfileAnalyzer,
    generate_synthetic_cohort_benchmarks,
)
from src.core.types import LongitudinalSessionPoint
from src.ui.charts import (
    render_affective_volatility_radar,
    render_longitudinal_recovery_gauge,
    render_longitudinal_trajectory_chart,
)


@pytest.fixture
def sample_profile():
    analyzer = LongitudinalProfileAnalyzer()
    points = [
        LongitudinalSessionPoint(
            session_id="s1",
            timestamp=1700000000.0,
            date_str="2026-08-01 10:00",
            mean_valence=-0.3,
            mean_arousal=0.2,
            engagement_score=50.0,
            attention_score=60.0,
            anomaly_count=2,
        ),
        LongitudinalSessionPoint(
            session_id="s2",
            timestamp=1700086400.0,
            date_str="2026-08-08 10:00",
            mean_valence=0.1,
            mean_arousal=0.1,
            engagement_score=70.0,
            attention_score=75.0,
            anomaly_count=0,
        ),
        LongitudinalSessionPoint(
            session_id="s3",
            timestamp=1700172800.0,
            date_str="2026-08-15 10:00",
            mean_valence=0.4,
            mean_arousal=0.15,
            engagement_score=85.0,
            attention_score=80.0,
            anomaly_count=0,
        ),
    ]
    return analyzer.analyze_profile("P_01", "Alex Vance", points)


def test_render_longitudinal_trajectory_chart(sample_profile):
    fig = render_longitudinal_trajectory_chart(sample_profile)
    assert fig is not None
    assert len(fig.data) >= 3  # Valence, Arousal, Engagement, Trend
    assert "Alex Vance" in fig.layout.title.text


def test_render_longitudinal_trajectory_empty():
    analyzer = LongitudinalProfileAnalyzer()
    empty_profile = analyzer.analyze_profile("P_00", "Empty", [])
    fig = render_longitudinal_trajectory_chart(empty_profile)
    assert fig is not None
    assert "No Longitudinal Trajectory" in fig.layout.title.text


def test_render_affective_volatility_radar(sample_profile):
    cohorts = generate_synthetic_cohort_benchmarks()
    fig = render_affective_volatility_radar(sample_profile, cohorts["Clinical Screening"])
    assert fig is not None
    assert len(fig.data) == 2  # Subject and Cohort


def test_render_longitudinal_recovery_gauge():
    fig = render_longitudinal_recovery_gauge(recovery_rate_sec=14.5, stability_score=82.0)
    assert fig is not None
    assert fig.data[0].value == 82.0
