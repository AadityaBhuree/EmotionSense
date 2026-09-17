"""Tests for Phase 7 Longitudinal Affective Profiling & Clinical Drift Analyzer."""

import pytest
from src.analytics.longitudinal_analyzer import (
    LongitudinalProfileAnalyzer,
    generate_synthetic_cohort_benchmarks,
)
from src.core.types import (
    CohortBenchmark,
    LongitudinalProfile,
    LongitudinalSessionPoint,
)


@pytest.fixture
def analyzer():
    return LongitudinalProfileAnalyzer()


@pytest.fixture
def sample_improving_points():
    return [
        LongitudinalSessionPoint(
            session_id="ses_01",
            timestamp=1700000000.0,
            date_str="2026-08-01",
            dominant_emotion="sadness",
            mean_valence=-0.40,
            mean_arousal=0.10,
            engagement_score=40.0,
            fatigue_score=60.0,
            anomaly_count=3,
        ),
        LongitudinalSessionPoint(
            session_id="ses_02",
            timestamp=1700086400.0,
            date_str="2026-08-08",
            dominant_emotion="neutral",
            mean_valence=-0.15,
            mean_arousal=0.05,
            engagement_score=55.0,
            fatigue_score=45.0,
            anomaly_count=1,
        ),
        LongitudinalSessionPoint(
            session_id="ses_03",
            timestamp=1700172800.0,
            date_str="2026-08-15",
            dominant_emotion="neutral",
            mean_valence=0.10,
            mean_arousal=0.00,
            engagement_score=68.0,
            fatigue_score=30.0,
            anomaly_count=0,
        ),
        LongitudinalSessionPoint(
            session_id="ses_04",
            timestamp=1700259200.0,
            date_str="2026-08-22",
            dominant_emotion="joy",
            mean_valence=0.35,
            mean_arousal=0.15,
            engagement_score=82.0,
            fatigue_score=15.0,
            anomaly_count=0,
        ),
    ]


@pytest.fixture
def sample_volatile_points():
    return [
        LongitudinalSessionPoint(
            session_id="ses_v1",
            timestamp=1700000000.0,
            date_str="2026-08-01",
            mean_valence=0.60,
            mean_arousal=0.70,
            anomaly_count=0,
        ),
        LongitudinalSessionPoint(
            session_id="ses_v2",
            timestamp=1700086400.0,
            date_str="2026-08-08",
            mean_valence=-0.70,
            mean_arousal=0.80,
            anomaly_count=4,
        ),
        LongitudinalSessionPoint(
            session_id="ses_v3",
            timestamp=1700172800.0,
            date_str="2026-08-15",
            mean_valence=0.55,
            mean_arousal=0.60,
            anomaly_count=0,
        ),
    ]


def test_empty_profile_handling(analyzer):
    profile = analyzer.analyze_profile("SUBJ_00", "Anonymous", [])
    assert isinstance(profile, LongitudinalProfile)
    assert profile.total_sessions == 0
    assert profile.first_session_date == ""
    assert profile.drift_metrics.trajectory_status == "STABLE_BASELINE"


def test_single_session_baseline(analyzer):
    p0 = LongitudinalSessionPoint(
        session_id="ses_00",
        timestamp=1700000000.0,
        date_str="2026-08-01",
        dominant_emotion="joy",
        mean_valence=0.45,
        anomaly_count=0,
    )
    profile = analyzer.analyze_profile("SUBJ_01", "Jane Doe", [p0])
    assert profile.total_sessions == 1
    assert profile.drift_metrics.valence_slope == 0.0
    assert profile.drift_metrics.stability_score == 100.0
    assert "Minimum 2 sessions required" in profile.drift_metrics.clinical_interpretation


def test_positive_clinical_progress_trajectory(analyzer, sample_improving_points):
    profile = analyzer.analyze_profile("SUBJ_02", "John Smith", sample_improving_points)
    assert profile.total_sessions == 4
    assert profile.first_session_date == "2026-08-01"
    assert profile.latest_session_date == "2026-08-22"

    metrics = profile.drift_metrics
    assert metrics.valence_slope > 0.05
    assert metrics.trajectory_status == "PROGRESSING_POSITIVELY"
    assert metrics.stability_score > 60.0
    assert len(metrics.recurrent_anomalies) == 2


def test_elevated_volatility_detection(analyzer, sample_volatile_points):
    profile = analyzer.analyze_profile("SUBJ_03", "Alice Ray", sample_volatile_points)
    metrics = profile.drift_metrics
    assert metrics.volatility_index >= 0.25
    assert metrics.trajectory_status == "ELEVATED_VOLATILITY"
    assert "Elevated emotional volatility" in metrics.clinical_interpretation


def test_cohort_percentile_calculation(analyzer, sample_improving_points):
    benchmarks = generate_synthetic_cohort_benchmarks()
    cohort = benchmarks["Clinical Screening"]
    assert isinstance(cohort, CohortBenchmark)

    profile = analyzer.analyze_profile("SUBJ_04", "Bob", sample_improving_points, cohort=cohort)
    assert 1.0 <= profile.cohort_percentile <= 99.0


def test_synthetic_cohort_benchmarks_generation():
    benchmarks = generate_synthetic_cohort_benchmarks()
    assert "Clinical Screening" in benchmarks
    assert "Talent Interview" in benchmarks
    assert "Wellness Tracking" in benchmarks
    assert benchmarks["Talent Interview"].sample_size > 500
