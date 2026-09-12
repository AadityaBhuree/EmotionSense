"""Dyadic Interaction Dynamics, Interpersonal Synchrony, and Rapport Modeling Engine.

Quantifies affective synchrony, facial mimicry, conversational floor balance, and interpersonal
rapport between two participants during live or recorded interactions (interviews, therapy, CX).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.core.types import DiarizationResult, DyadicInteractionMetrics
from src.utils.logger import get_logger

logger = get_logger("DyadicInteractionAnalyzer")


class DyadicInteractionAnalyzer:
    """Computes interpersonal synchrony, conversational balance, and Dyadic Rapport Score."""

    def __init__(
        self,
        synchrony_window_sec: float = 30.0,
        mimicry_lag_window_sec: float = 2.0,
        sample_rate_hz: float = 2.0,
    ):
        """Initializes the dyadic analyzer.

        Args:
            synchrony_window_sec: Default temporal sliding window for affective correlation.
            mimicry_lag_window_sec: Maximum reaction lag for facial mimicry cross-correlation.
            sample_rate_hz: Common sampling rate for aligning multi-subject timelines.
        """
        self.synchrony_window_sec = synchrony_window_sec
        self.mimicry_lag_window_sec = mimicry_lag_window_sec
        self.sample_rate_hz = sample_rate_hz

    def analyze(
        self,
        participant_a_samples: List[Dict[str, Any]],
        participant_b_samples: List[Dict[str, Any]],
        diarization: Optional[DiarizationResult] = None,
    ) -> DyadicInteractionMetrics:
        """Analyzes multi-subject temporal affect data and optional speech diarization.

        Args:
            participant_a_samples: Time-series sample dictionaries for Participant A.
                Expected keys: 'timestamp' (or 'time'), 'valence', 'arousal', 'smile' (or 'au12'),
                optional 'yaw', 'pitch'.
            participant_b_samples: Time-series sample dictionaries for Participant B.
            diarization: Optional DiarizationResult containing speaker turns and durations.

        Returns:
            DyadicInteractionMetrics object with computed synchrony, rapport, and balance.
        """
        if not participant_a_samples or not participant_b_samples:
            return self._default_metrics("Insufficient participant samples for dyadic analysis.")

        # 1. Align time-series onto common grid
        times_a, val_a, aro_a, smile_a, yaw_a = self._extract_series(participant_a_samples)
        times_b, val_b, aro_b, smile_b, yaw_b = self._extract_series(participant_b_samples)

        if len(val_a) < 2 or len(val_b) < 2:
            return self._default_metrics("Fewer than 2 valid temporal samples.")

        t_min = max(times_a[0], times_b[0])
        t_max = min(times_a[-1], times_b[-1])
        duration = t_max - t_min

        if duration <= 0.5:
            return self._default_metrics("Insufficient temporal overlap between participants.")

        # Resample onto uniform grid
        dt = 1.0 / self.sample_rate_hz
        grid = np.arange(t_min, t_max, dt)
        if len(grid) < 2:
            return self._default_metrics("Grid resolution too coarse for overlap duration.")

        grid_val_a = np.interp(grid, times_a, val_a)
        grid_val_b = np.interp(grid, times_b, val_b)
        grid_aro_a = np.interp(grid, times_a, aro_a)
        grid_aro_b = np.interp(grid, times_b, aro_b)
        grid_smile_a = np.interp(grid, times_a, smile_a)
        grid_smile_b = np.interp(grid, times_b, smile_b)
        grid_yaw_a = np.interp(grid, times_a, yaw_a)
        grid_yaw_b = np.interp(grid, times_b, yaw_b)

        # 2. Compute Affective Synchrony (Pearson cross-correlation)
        val_sync = self._compute_correlation(grid_val_a, grid_val_b)
        aro_sync = self._compute_correlation(grid_aro_a, grid_aro_b)

        # 3. Compute Facial Mimicry Index (Lagged cross-correlation of smiles)
        mimicry_idx = self._compute_mimicry(grid_smile_a, grid_smile_b, dt)

        # 4. Mutual Attentiveness (Head Pose orientation)
        attentiveness = self._compute_attentiveness(grid_yaw_a, grid_yaw_b)

        # 5. Conversational Floor Balance & Turn Metrics (from Diarization)
        conv_balance, dom_speaker, trans_lat, int_rate = self._compute_speech_metrics(diarization, duration)

        # 6. Dyadic Rapport Score Calculation (0.0 to 100.0)
        # Weights: Valence Synchrony (25%), Arousal Synchrony (10%), Floor Balance (20%), Mimicry (20%), Attentiveness & Low Friction (25%)
        norm_val_sync = max(0.0, min(1.0, (val_sync + 1.0) / 2.0))
        norm_aro_sync = max(0.0, min(1.0, (aro_sync + 1.0) / 2.0))

        rapport = (
            norm_val_sync * 25.0
            + norm_aro_sync * 10.0
            + conv_balance * 20.0
            + mimicry_idx * 20.0
            + attentiveness * 15.0
            + max(0.0, 10.0 - int_rate * 2.0)
        )
        rapport = round(float(np.clip(rapport, 0.0, 100.0)), 1)

        # 7. Qualitative Resonance Categorization & Notes
        category, notes = self._synthesize_resonance(
            rapport, val_sync, aro_sync, conv_balance, dom_speaker, mimicry_idx, int_rate
        )

        return DyadicInteractionMetrics(
            rapport_score=rapport,
            valence_synchrony=round(val_sync, 3),
            arousal_synchrony=round(aro_sync, 3),
            mimicry_index=round(mimicry_idx, 3),
            conversational_balance=round(conv_balance, 3),
            dominance_speaker=dom_speaker,
            mutual_attentiveness=round(attentiveness, 3),
            turn_transition_latency=round(trans_lat, 2),
            interruption_rate=round(int_rate, 2),
            resonance_category=category,
            summary_notes=notes,
        )

    def _extract_series(
        self, samples: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extracts sorted numpy arrays for timestamp, valence, arousal, smile, and yaw."""
        # Sort by timestamp
        sorted_samples = sorted(
            samples, key=lambda s: s.get("timestamp", s.get("time", 0.0))
        )

        times = []
        vals = []
        aros = []
        smiles = []
        yaws = []

        for s in sorted_samples:
            t = float(s.get("timestamp", s.get("time", 0.0)))
            # Try direct valence or nested affect dictionary
            affect = s.get("affect", {})
            if isinstance(affect, dict):
                v = float(affect.get("valence", s.get("valence", 0.0)))
                a = float(affect.get("arousal", s.get("arousal", 0.0)))
            else:
                v = float(getattr(affect, "valence", s.get("valence", 0.0)))
                a = float(getattr(affect, "arousal", s.get("arousal", 0.0)))

            # Smile AU12 intensity
            aus = s.get("action_units", {})
            if isinstance(aus, dict):
                sm = float(aus.get("lip_corner_puller", s.get("smile", 0.0)))
            else:
                sm = float(getattr(aus, "lip_corner_puller", s.get("smile", 0.0)))

            # Head pose yaw
            pose = s.get("head_pose", {})
            if isinstance(pose, dict):
                y = float(pose.get("yaw", s.get("yaw", 0.0)))
            else:
                y = float(getattr(pose, "yaw", s.get("yaw", 0.0)))

            times.append(t)
            vals.append(v)
            aros.append(a)
            smiles.append(sm)
            yaws.append(y)

        return (
            np.array(times, dtype=np.float64),
            np.array(vals, dtype=np.float32),
            np.array(aros, dtype=np.float32),
            np.array(smiles, dtype=np.float32),
            np.array(yaws, dtype=np.float32),
        )

    def _compute_correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        """Computes Pearson correlation coefficient with standard deviation guarding."""
        std_a = np.std(a)
        std_b = np.std(b)
        if std_a < 1e-4 or std_b < 1e-4:
            # If both have near-constant matching values, return positive alignment
            diff = np.abs(np.mean(a) - np.mean(b))
            return max(0.0, 1.0 - diff) if diff < 0.3 else 0.0

        r = np.corrcoef(a, b)[0, 1]
        return float(np.nan_to_num(r, nan=0.0))

    def _compute_mimicry(self, smile_a: np.ndarray, smile_b: np.ndarray, dt: float) -> float:
        """Computes lagged cross-correlation within human mimicry window (0.5s - 2.5s)."""
        max_lag_samples = int(self.mimicry_lag_window_sec / dt)
        if max_lag_samples < 1 or len(smile_a) < max_lag_samples * 2:
            # Fallback to direct correlation
            return max(0.0, self._compute_correlation(smile_a, smile_b))

        best_corr = 0.0
        # Positive lags (A leads, B mimics) and negative lags (B leads, A mimics)
        for lag in range(1, max_lag_samples + 1):
            corr_ab = self._compute_correlation(smile_a[:-lag], smile_b[lag:])
            corr_ba = self._compute_correlation(smile_b[:-lag], smile_a[lag:])
            best_corr = max(best_corr, corr_ab, corr_ba)

        return float(np.clip(best_corr, 0.0, 1.0))

    def _compute_attentiveness(self, yaw_a: np.ndarray, yaw_b: np.ndarray) -> float:
        """Computes mutual visual orientation attentiveness."""
        # Extreme yaw angles (> 25 degrees) indicate looking away from conversational partner/camera
        mean_abs_yaw_a = float(np.mean(np.abs(yaw_a)))
        mean_abs_yaw_b = float(np.mean(np.abs(yaw_b)))

        score_a = max(0.0, 1.0 - (mean_abs_yaw_a / 35.0))
        score_b = max(0.0, 1.0 - (mean_abs_yaw_b / 35.0))
        return float((score_a + score_b) / 2.0)

    def _compute_speech_metrics(
        self, diarization: Optional[DiarizationResult], session_duration: float
    ) -> Tuple[float, str, float, float]:
        """Calculates conversational floor balance, dominant speaker, transition latency, and interruption rate."""
        if not diarization or not diarization.speaker_durations:
            # Neutral default when no speech data is present
            return 1.0, "balanced", 0.4, 0.0

        durs = list(diarization.speaker_durations.values())
        speakers = list(diarization.speaker_durations.keys())

        if len(durs) >= 2:
            d_a = durs[0]
            d_b = durs[1]
            total_speech = d_a + d_b
            if total_speech > 0.0:
                balance = 1.0 - abs(d_a - d_b) / total_speech
                ratio_a = d_a / total_speech
                ratio_b = d_b / total_speech
                if ratio_a > 0.65:
                    dom = speakers[0]
                elif ratio_b > 0.65:
                    dom = speakers[1]
                else:
                    dom = "balanced"
            else:
                balance = 1.0
                dom = "balanced"
        else:
            balance = 0.0
            dom = speakers[0] if speakers else "Speaker_0"

        # Turn transition latency
        turns = diarization.turns
        latencies: List[float] = []
        for i in range(1, len(turns)):
            gap = turns[i].start_time - turns[i - 1].end_time
            if 0.0 <= gap <= 3.0:
                latencies.append(gap)

        avg_latency = float(np.mean(latencies)) if latencies else 0.5

        # Interruption rate per minute
        duration_minutes = max(0.1, session_duration / 60.0)
        int_rate = diarization.interruption_count / duration_minutes

        return float(np.clip(balance, 0.0, 1.0)), dom, avg_latency, int_rate

    def _synthesize_resonance(
        self,
        rapport: float,
        val_sync: float,
        aro_sync: float,
        balance: float,
        dom_speaker: str,
        mimicry: float,
        int_rate: float,
    ) -> Tuple[str, List[str]]:
        """Synthesizes clinical and interview resonance category with qualitative diagnostic notes."""
        notes = []

        # Category determination
        if rapport >= 75.0 and balance >= 0.60:
            category = "High Collaborative Resonance"
            notes.append("Strong mutual affective alignment with balanced conversational engagement.")
        elif val_sync > 0.45:
            category = "Empathic / Supportive"
            notes.append("High valence synchronization indicating active empathy and emotional attunement.")
        elif dom_speaker != "balanced" and balance < 0.40:
            category = "Asymmetric / Dominant"
            notes.append(f"Conversational floor heavily skewed toward {dom_speaker} ({round((1-balance)*100)}% imbalance).")
        elif val_sync < -0.25 or int_rate > 3.0:
            category = "Tense / Dissonant"
            notes.append("Elevated conversational friction with discordant affect trajectories or high interruption frequency.")
        elif aro_sync < 0.1 and mimicry < 0.15:
            category = "Passive / Guarded"
            notes.append("Low affective contagion and restrained micro-expression responsiveness.")
        else:
            category = "Neutral / Moderately Attuned"
            notes.append("Steady, conventional conversational flow with moderate rapport.")

        # Additional diagnostic observations
        if mimicry >= 0.4:
            notes.append(f"Pronounced facial smile mimicry detected (Index {round(mimicry, 2)}), signaling genuine rapport.")
        elif mimicry < 0.1:
            notes.append("Minimal facial mimicry observed; participant maintained reserved facial composure.")

        if int_rate > 2.5:
            notes.append(f"Frequent turn collision/interruption rate: {round(int_rate, 1)} interruptions per minute.")

        if balance >= 0.85:
            notes.append("Exemplary conversational floor distribution between participants (~50/50 balance).")

        return category, notes

    def _default_metrics(self, reason: str) -> DyadicInteractionMetrics:
        """Returns baseline neutral metrics when data is insufficient."""
        return DyadicInteractionMetrics(
            rapport_score=50.0,
            valence_synchrony=0.0,
            arousal_synchrony=0.0,
            mimicry_index=0.0,
            conversational_balance=1.0,
            dominance_speaker="balanced",
            mutual_attentiveness=0.5,
            resonance_category="Insufficient Data",
            summary_notes=[reason],
        )
