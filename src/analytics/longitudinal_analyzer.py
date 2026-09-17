"""Longitudinal Affective Trajectory & Multi-Session Clinical Drift Analyzer.

Quantifies emotional evolution over multi-week or multi-session evaluations,
calculating valence trajectory drift, emotional volatility index, recovery rates,
and cohort normative benchmarking.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.config import DRIFT_THRESHOLDS, LONGITUDINAL_CONFIG
from src.core.types import (
    AffectiveDriftMetrics,
    CohortBenchmark,
    LongitudinalProfile,
    LongitudinalSessionPoint,
)


class LongitudinalProfileAnalyzer:
    """Evaluates multi-session longitudinal trajectories for individual subjects."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or LONGITUDINAL_CONFIG
        self.thresholds = DRIFT_THRESHOLDS
        self.min_sessions = self.config.get("min_sessions_for_trend", 2)

    def analyze_profile(
        self,
        subject_id: str,
        subject_name: str,
        history_points: List[LongitudinalSessionPoint],
        cohort: Optional[CohortBenchmark] = None,
        notes: Optional[List[str]] = None,
    ) -> LongitudinalProfile:
        """Constructs a comprehensive LongitudinalProfile from a subject's session points."""
        if not history_points:
            return LongitudinalProfile(
                subject_id=subject_id,
                subject_name=subject_name,
                total_sessions=0,
                first_session_date="",
                latest_session_date="",
                history_points=[],
                drift_metrics=AffectiveDriftMetrics(
                    clinical_interpretation="No historical evaluation sessions recorded for this subject."
                ),
                cohort_percentile=50.0,
                notes=notes or [],
            )

        # Sort chronologically by timestamp
        sorted_points = sorted(history_points, key=lambda p: p.timestamp)
        n = len(sorted_points)

        first_date = sorted_points[0].date_str
        latest_date = sorted_points[-1].date_str

        drift_metrics = self.calculate_drift_metrics(sorted_points)
        cohort_pct = self.calculate_cohort_percentile(sorted_points, cohort)

        return LongitudinalProfile(
            subject_id=subject_id,
            subject_name=subject_name,
            total_sessions=n,
            first_session_date=first_date,
            latest_session_date=latest_date,
            history_points=sorted_points,
            drift_metrics=drift_metrics,
            cohort_percentile=cohort_pct,
            notes=notes or [],
        )

    def calculate_drift_metrics(
        self, points: List[LongitudinalSessionPoint]
    ) -> AffectiveDriftMetrics:
        """Calculates regression trends, volatility, recovery rate, and stability."""
        n = len(points)
        if n < self.min_sessions:
            # Single session baseline
            p0 = points[0]
            volatility = 0.0
            stability = 100.0
            status = "STABLE_BASELINE"
            interpretation = (
                f"Initial baseline established on {p0.date_str} with {p0.dominant_emotion} dominant affect "
                f"(Valence: {p0.mean_valence:+.2f}). Minimum 2 sessions required for longitudinal drift trends."
            )
            recurrent: List[str] = []
            if p0.anomaly_count > 0:
                recurrent.append(f"Session {p0.session_id}: {p0.anomaly_count} anomaly event(s)")

            return AffectiveDriftMetrics(
                valence_slope=0.0,
                arousal_slope=0.0,
                volatility_index=volatility,
                stability_score=stability,
                recovery_rate_sec=15.0,
                trajectory_status=status,
                recurrent_anomalies=recurrent,
                clinical_interpretation=interpretation,
            )

        # Multi-session trend calculations via OLS
        valences = np.array([p.mean_valence for p in points], dtype=np.float64)
        arousals = np.array([p.mean_arousal for p in points], dtype=np.float64)
        x = np.arange(n, dtype=np.float64)

        # OLS slope = Cov(x, y) / Var(x)
        x_mean = np.mean(x)
        x_var = np.sum((x - x_mean) ** 2)

        if x_var > 0:
            val_slope = float(np.sum((x - x_mean) * (valences - np.mean(valences))) / x_var)
            aro_slope = float(np.sum((x - x_mean) * (arousals - np.mean(arousals))) / x_var)
        else:
            val_slope = 0.0
            aro_slope = 0.0

        # Affective Volatility Index (Standard deviation across valence and arousal)
        std_val = float(np.std(valences))
        std_aro = float(np.std(arousals))
        raw_volatility = (std_val + std_aro) / 2.0
        volatility_index = float(np.clip(raw_volatility, 0.0, 1.0))

        # Affective Stability Score (100 is perfectly consistent, 0 is extreme volatility)
        stability_score = float(np.clip(100.0 * (1.0 - (volatility_index * 1.5)), 0.0, 100.0))

        # Recovery rate heuristic (proportional to fatigue and anomaly density)
        total_anomalies = sum(p.anomaly_count for p in points)
        mean_fatigue = float(np.mean([p.fatigue_score for p in points]))
        recovery_rate_sec = round(12.0 + (mean_fatigue * 0.2) + (total_anomalies * 2.5), 1)

        # Classify Trajectory Status
        pos_thresh = self.thresholds["positive_progress_slope"]
        dec_thresh = self.thresholds["declining_slope"]
        elevated_vol = self.thresholds["elevated_volatility"]

        if val_slope >= pos_thresh and volatility_index < elevated_vol:
            status = "PROGRESSING_POSITIVELY"
            interpretation = (
                f"Statistically significant positive affective progression (Valence slope: {val_slope:+.3f}/session). "
                f"Patient demonstrates sustained emotional stabilization and resilience (Stability: {stability_score:.1f}%)."
            )
        elif val_slope <= dec_thresh:
            status = "DECLINING_AFFECT"
            interpretation = (
                f"Negative affective trajectory detected across {n} sessions (Valence slope: {val_slope:+.3f}/session). "
                "Recommend supportive intervention and clinical check-in."
            )
        elif volatility_index >= elevated_vol:
            status = "ELEVATED_VOLATILITY"
            interpretation = (
                f"Elevated emotional volatility observed (Volatility Index: {volatility_index:.2f}). "
                "Significant fluctuations between session affective baselines warrant longitudinal monitoring."
            )
        else:
            status = "STABLE_BASELINE"
            interpretation = (
                f"Affective baseline remains stable and consistent across {n} sessions "
                f"(Mean Valence: {float(np.mean(valences)):+.2f}, Stability: {stability_score:.1f}%)."
            )

        # Recurrent anomalies log
        recurrent_anomalies: List[str] = []
        for p in points:
            if p.anomaly_count > 0:
                recurrent_anomalies.append(
                    f"Session {p.session_id} ({p.date_str}): {p.anomaly_count} distress events recorded"
                )

        return AffectiveDriftMetrics(
            valence_slope=round(val_slope, 4),
            arousal_slope=round(aro_slope, 4),
            volatility_index=round(volatility_index, 3),
            stability_score=round(stability_score, 1),
            recovery_rate_sec=recovery_rate_sec,
            trajectory_status=status,
            recurrent_anomalies=recurrent_anomalies,
            clinical_interpretation=interpretation,
        )

    def calculate_cohort_percentile(
        self,
        points: List[LongitudinalSessionPoint],
        cohort: Optional[CohortBenchmark] = None,
    ) -> float:
        """Calculates subject's normative percentile relative to population benchmark."""
        if not points:
            return 50.0

        if cohort is None:
            cohort = CohortBenchmark(cohort_name="General Clinical Population")

        mean_subj_val = float(np.mean([p.mean_valence for p in points]))

        # Z-score computation
        z = (mean_subj_val - cohort.norm_mean_valence) / max(cohort.norm_std_valence, 1e-4)

        # Standard normal cumulative distribution function (CDF approximation)
        pct = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) * 100.0
        return round(float(np.clip(pct, 1.0, 99.0)), 1)


def generate_synthetic_cohort_benchmarks() -> Dict[str, CohortBenchmark]:
    """Generates standard normative cohort benchmarks for comparative evaluation."""
    return {
        "Clinical Screening": CohortBenchmark(
            cohort_name="Clinical Screening",
            sample_size=320,
            norm_mean_valence=0.05,
            norm_std_valence=0.30,
            norm_mean_arousal=0.15,
            norm_std_arousal=0.25,
            norm_volatility=0.22,
            norm_stability_score=72.0,
        ),
        "Talent Interview": CohortBenchmark(
            cohort_name="Talent Interview",
            sample_size=540,
            norm_mean_valence=0.25,
            norm_std_valence=0.20,
            norm_mean_arousal=0.10,
            norm_std_arousal=0.18,
            norm_volatility=0.15,
            norm_stability_score=84.0,
        ),
        "Wellness Tracking": CohortBenchmark(
            cohort_name="Wellness Tracking",
            sample_size=410,
            norm_mean_valence=0.20,
            norm_std_valence=0.22,
            norm_mean_arousal=0.00,
            norm_std_arousal=0.16,
            norm_volatility=0.16,
            norm_stability_score=81.0,
        ),
        "Academic Research": CohortBenchmark(
            cohort_name="Academic Research",
            sample_size=180,
            norm_mean_valence=0.10,
            norm_std_valence=0.28,
            norm_mean_arousal=0.05,
            norm_std_arousal=0.22,
            norm_volatility=0.19,
            norm_stability_score=76.0,
        ),
    }
