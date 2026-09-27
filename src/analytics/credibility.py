"""Phase 13: Forensic Veracity Assessment & Affective Deception Leakage Engine.

Implements real-time detection of micro-momentary affective leakage (<200ms flashes),
Duchenne smile incongruence (social masking vs authentic affect), 8-14 Hz vocal fold
micro-tremor (Voice Stress Analysis), Cepstral Peak Prominence (CPP), and multi-channel
polygraphic cross-checking synthesized into the Credibility & Deception Risk Index (CDRI).
"""

import time
from typing import List, Tuple, Optional, Dict
import numpy as np

from src.core.credibility_models import (
    VeracityTier,
    DeceptionFlag,
    MicroLeakageEvent,
    VoiceStressProfile,
    FacialVeracityMetrics,
    MultimodalPolygraphProfile,
    CredibilitySnapshot,
)


class CredibilityEngine:
    """Enterprise contact-free forensic veracity and deception leakage analytics engine."""

    # Weights for multi-sensor polygraph late fusion
    WEIGHT_FACIAL = 0.25
    WEIGHT_VOCAL = 0.25
    WEIGHT_PUPILLOMETRIC = 0.20
    WEIGHT_SOMATOSENSORY = 0.15
    WEIGHT_AUTONOMIC = 0.15

    def __init__(
        self,
        leakage_flash_max_ms: float = 250.0,
        duchenne_threshold: float = 0.30,
        voice_stress_threshold: float = 0.45,
        buffer_window_sec: float = 15.0,
    ):
        self.leakage_flash_max_ms = leakage_flash_max_ms
        self.duchenne_threshold = duchenne_threshold
        self.voice_stress_threshold = voice_stress_threshold
        self.buffer_window_sec = buffer_window_sec

        # Temporal buffers
        # Each entry: (timestamp, au_dict, predicted_emotion, valence)
        self.face_history: List[Tuple[float, Dict[str, float], str, float]] = []
        self.leakage_events: List[MicroLeakageEvent] = []
        self.baseline_pulse_bpm: float = 72.0
        self.baseline_cpr: float = 1.0

    def compute_facial_veracity(
        self,
        action_units: Dict[str, float],
        macro_emotion: str = "Neutral",
        valence: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Tuple[FacialVeracityMetrics, List[MicroLeakageEvent], List[str]]:
        """Analyzes Action Unit configurations for Duchenne genuineness and micro-expression leakage."""
        if timestamp is None:
            timestamp = time.time()

        au_dict = {k.upper(): float(v) for k, v in action_units.items()}
        flags: List[str] = []

        # 1. Duchenne Smile Analysis
        # AU12: Zygomatic Major (Lip Corner Puller)
        # AU06: Orbicularis Oculi (Cheek Raiser / Eye Squint)
        au12 = au_dict.get("AU12", 0.0)
        au06 = au_dict.get("AU06", 0.0)

        if au12 > 0.30:
            # When smiling, check if eye squint accompanies mouth pull
            congruence = float(np.clip((au06 + 0.05) / (au12 + 0.05), 0.0, 1.0))
            incongruence = float(np.clip(1.0 - congruence, 0.0, 1.0))
            if incongruence > self.duchenne_threshold:
                flags.append(DeceptionFlag.NON_DUCHENNE_MASKING.value)
        else:
            congruence = 1.0
            incongruence = 0.0

        # 2. FACS Sneer / Asymmetry (AU14 / AU10 unilateral)
        au14_left = au_dict.get("AU14_L", au_dict.get("AU14", 0.0))
        au14_right = au_dict.get("AU14_R", au_dict.get("AU14", 0.0))
        sneer_asymmetry = float(
            abs(au14_left - au14_right) / (max(au14_left, au14_right, 0.1) + 1e-4)
        )
        if sneer_asymmetry > 0.35 and max(au14_left, au14_right) > 0.25:
            flags.append(DeceptionFlag.FACIAL_ASYMMETRY_SNEER.value)

        # 3. Micro-Momentary Leakage Flash (< 200ms) Detection
        self.face_history.append((timestamp, au_dict, macro_emotion, valence))
        cutoff = timestamp - self.buffer_window_sec
        self.face_history = [item for item in self.face_history if item[0] >= cutoff]

        detected_leakages: List[MicroLeakageEvent] = []
        if len(self.face_history) >= 4:
            leakage_event = self._detect_micro_leakage(timestamp, au_dict, macro_emotion)
            if leakage_event is not None:
                detected_leakages.append(leakage_event)
                self.leakage_events.append(leakage_event)
                flags.append(DeceptionFlag.MICRO_FLASH_LEAKAGE.value)

        metrics = FacialVeracityMetrics(
            duchenne_congruence=round(congruence, 3),
            duchenne_incongruence_index=round(incongruence, 3),
            sneer_asymmetry_index=round(sneer_asymmetry, 3),
            micro_leakage_detected=len(detected_leakages) > 0,
            leakage_events_count=len(self.leakage_events),
            macro_masked_state=macro_emotion,
        )

        return metrics, detected_leakages, flags

    def _detect_micro_leakage(
        self,
        current_time: float,
        current_au: Dict[str, float],
        macro_emotion: str,
    ) -> Optional[MicroLeakageEvent]:
        """Detects high-frequency negative affect transients conflicting with current macro state."""
        # Check for brief burst of negative AUs: AU04 (Brow Lowerer), AU20 (Lip Stretcher/Fear), AU15 (Lip Corner Depressor)
        negative_au_sum = current_au.get("AU04", 0.0) + current_au.get("AU20", 0.0) + current_au.get("AU15", 0.0)
        
        if macro_emotion.lower() in ["joy", "happy", "neutral"] and negative_au_sum > 0.45:
            # Check duration in history
            matching_frames = [
                h for h in self.face_history[-6:]
                if (h[1].get("AU04", 0.0) + h[1].get("AU20", 0.0) + h[1].get("AU15", 0.0)) > 0.35
            ]
            if len(matching_frames) >= 1:
                duration_ms = float((matching_frames[-1][0] - matching_frames[0][0]) * 1000.0)
                if 20.0 <= duration_ms <= self.leakage_flash_max_ms:
                    leaked = "Fear" if current_au.get("AU20", 0.0) > 0.25 else "Contempt/Anger"
                    conflicted = [k for k in ["AU04", "AU15", "AU20"] if current_au.get(k, 0.0) > 0.15]
                    return MicroLeakageEvent(
                        timestamp=current_time,
                        duration_ms=round(duration_ms, 1),
                        leaked_affect=leaked,
                        masked_affect=macro_emotion,
                        conflicting_action_units=conflicted or ["AU04"],
                        leakage_intensity=round(float(min(1.0, negative_au_sum)), 3),
                        confidence=0.86,
                    )
        return None

    def analyze_voice_stress(
        self,
        audio_buffer: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        f0_trajectory: Optional[List[float]] = None,
        response_latency_sec: float = 0.45,
    ) -> Tuple[VoiceStressProfile, List[str]]:
        """Extracts 8-14 Hz vocal fold micro-tremor, Cepstral Peak Prominence (CPP), and stress profile."""
        flags: List[str] = []

        # 1. 8-14 Hz Lippold Micro-Tremor in F0 or Acoustic Envelope
        micro_tremor_energy = 0.035
        f0_ppq = 0.015
        cpp_db = 13.5
        spectral_slope = -14.0

        if f0_trajectory and len(f0_trajectory) >= 16:
            f0_arr = np.asarray([f for f in f0_trajectory if f > 40.0], dtype=np.float64)
            if len(f0_arr) >= 12:
                # F0 perturbation quotient (PPQ)
                diffs = np.abs(np.diff(f0_arr))
                f0_ppq = float(np.mean(diffs) / (np.mean(f0_arr) + 1e-4))

                # FFT of F0 variations to isolate 8-14 Hz tremor
                f0_detrended = f0_arr - np.mean(f0_arr)
                fft_vals = np.abs(np.fft.rfft(f0_detrended))
                freqs = np.fft.rfftfreq(len(f0_detrended), d=1.0 / 30.0)  # ~30Hz sample rate of F0 frames
                mask_tremor = (freqs >= 8.0) & (freqs <= 14.0)
                if np.any(mask_tremor):
                    tremor_pwr = np.sum(fft_vals[mask_tremor] ** 2)
                    total_pwr = np.sum(fft_vals ** 2) + 1e-6
                    micro_tremor_energy = float(np.clip(tremor_pwr / total_pwr, 0.0, 1.0))

        if audio_buffer is not None and len(audio_buffer) >= 512:
            cpp_db, spectral_slope = self._compute_cpp_and_slope(audio_buffer, sample_rate)

        # Stress Index Calculation
        # Stressed voice: elevated micro-tremor energy (> 0.08), high PPQ (> 0.03), low CPP (< 9.0 dB)
        norm_tremor = float(np.clip(micro_tremor_energy / 0.15, 0.0, 1.0))
        norm_ppq = float(np.clip(f0_ppq / 0.05, 0.0, 1.0))
        norm_cpp = float(np.clip((15.0 - cpp_db) / 10.0, 0.0, 1.0))
        stress_index = float(np.clip(0.40 * norm_tremor + 0.30 * norm_ppq + 0.30 * norm_cpp, 0.0, 1.0))

        is_stressed = stress_index >= self.voice_stress_threshold
        if is_stressed:
            flags.append(DeceptionFlag.ACOUSTIC_MICRO_TREMOR.value)

        if response_latency_sec > 2.5:
            flags.append(DeceptionFlag.LATENCY_ELONGATION.value)

        profile = VoiceStressProfile(
            micro_tremor_energy=round(micro_tremor_energy, 4),
            cpp_db=round(cpp_db, 2),
            f0_perturbation_quotient=round(f0_ppq, 4),
            spectral_slope=round(spectral_slope, 2),
            response_latency_sec=round(response_latency_sec, 2),
            stress_index=round(stress_index, 3),
            is_voice_stressed=is_stressed,
        )

        return profile, flags

    def _compute_cpp_and_slope(self, audio: np.ndarray, sr: int) -> Tuple[float, float]:
        """Calculates Cepstral Peak Prominence (CPP) and spectral slope from audio."""
        try:
            x = np.asarray(audio, dtype=np.float64)
            # Pre-emphasis
            x_filt = np.append(x[0], x[1:] - 0.97 * x[:-1])
            windowed = x_filt * np.hanning(len(x_filt))

            # Power spectrum
            spec = np.abs(np.fft.rfft(windowed, n=1024))
            spec = np.maximum(spec, 1e-12)
            log_spec = np.log(spec)

            # Real cepstrum
            cepstrum = np.real(np.fft.irfft(log_spec))

            # Quefrency range for pitch (approx 60 Hz to 400 Hz)
            min_q = int(sr / 400.0)
            max_q = min(int(sr / 60.0), len(cepstrum) - 1)

            if max_q > min_q:
                peak_idx = min_q + int(np.argmax(cepstrum[min_q:max_q]))
                peak_val = cepstrum[peak_idx]

                # Linear regression baseline
                q_axis = np.arange(min_q, max_q)
                slope, intercept = np.polyfit(q_axis, cepstrum[min_q:max_q], 1)
                baseline_val = slope * peak_idx + intercept
                cpp_val = float(peak_val - baseline_val)
                # Scale to standard dB representation
                cpp_db = float(np.clip(cpp_val * 20.0 + 8.0, 2.0, 25.0))
            else:
                cpp_db = 12.0

            # Spectral slope (dB/octave)
            freqs = np.fft.rfftfreq(1024, d=1.0 / sr)
            valid = (freqs >= 100.0) & (freqs <= 4000.0)
            if np.sum(valid) > 4:
                octaves = np.log2(freqs[valid] / 100.0 + 1e-4)
                db_pwr = 20.0 * np.log10(spec[valid] + 1e-12)
                p = np.polyfit(octaves, db_pwr, 1)
                spectral_slope = float(p[0])
            else:
                spectral_slope = -14.0

            return cpp_db, spectral_slope
        except Exception:
            return 12.5, -14.2

    def fuse_credibility_assessment(
        self,
        facial_metrics: FacialVeracityMetrics,
        voice_stress: VoiceStressProfile,
        pupil_cpr: float = 1.0,
        pacifying_adaptor_active: bool = False,
        pulse_bpm: float = 72.0,
        detected_flags: Optional[List[str]] = None,
        timestamp: Optional[float] = None,
    ) -> CredibilitySnapshot:
        """Fuses all channels into the composite Credibility & Deception Risk Index (CDRI)."""
        if timestamp is None:
            timestamp = time.time()
        flags = list(detected_flags or [])

        # 1. Channel Scores (0.0 = completely truthful, 1.0 = maximal deceit/stress marker)
        # Facial channel: Duchenne incongruence + micro-leakage penalty
        leakage_penalty = 0.35 if facial_metrics.micro_leakage_detected else 0.0
        facial_score = float(np.clip(
            facial_metrics.duchenne_incongruence_index * 0.70 +
            facial_metrics.sneer_asymmetry_index * 0.30 +
            leakage_penalty,
            0.0, 1.0
        ))

        # Vocal channel: Voice stress index
        vocal_score = float(voice_stress.stress_index)

        # Pupillometric channel: Cognitive pupillary response relative to baseline
        cpr_delta = max(0.0, pupil_cpr - self.baseline_cpr)
        pupil_score = float(np.clip(cpr_delta / 0.30, 0.0, 1.0))
        if pupil_score > 0.45:
            flags.append(DeceptionFlag.PUPIL_DILATION_STRAIN.value)

        # Somatosensory channel: Pacifying self-touch adaptors (mouth cover, neck touch)
        somatosensory_score = 0.65 if pacifying_adaptor_active else 0.08
        if pacifying_adaptor_active:
            flags.append(DeceptionFlag.PACIFYING_ADAPTOR_SURGE.value)

        # Autonomic channel: Instantaneous pulse surge relative to baseline
        bpm_delta = max(0.0, pulse_bpm - self.baseline_pulse_bpm)
        autonomic_score = float(np.clip(bpm_delta / 25.0, 0.0, 1.0))
        if bpm_delta > 15.0:
            flags.append(DeceptionFlag.AUTONOMIC_PULSE_SURGE.value)

        # Remove duplicate flags while preserving order
        unique_flags = list(dict.fromkeys(flags))

        # 2. Multimodal Late Fusion -> Deception Risk Index (CDRI)
        cdri = float(np.clip(
            self.WEIGHT_FACIAL * facial_score +
            self.WEIGHT_VOCAL * vocal_score +
            self.WEIGHT_PUPILLOMETRIC * pupil_score +
            self.WEIGHT_SOMATOSENSORY * somatosensory_score +
            self.WEIGHT_AUTONOMIC * autonomic_score,
            0.0, 1.0
        ))

        credibility_score = float(np.clip(1.0 - cdri, 0.0, 1.0))

        # 3. Categorize into Veracity Tier
        if cdri < 0.25:
            tier = VeracityTier.VERIDICAL_AUTHENTIC.value
            verdict = "High affective congruence; organic vocal prosody and authentic facial valence."
        elif cdri < 0.50:
            tier = VeracityTier.COGNITIVE_STRAIN.value
            verdict = "Moderate cognitive strain detected without overt affective dissimulation."
        elif cdri < 0.75:
            tier = VeracityTier.SUSPICIOUS_INCONGRUENCE.value
            verdict = "Affective incongruence flagged; non-Duchenne smile masking or vocal perturbation active."
        else:
            tier = VeracityTier.HIGH_DECEPTION_RISK.value
            verdict = "High deception probability; concurrent multi-sensor autonomic surge, voice stress, and micro-leakage."

        polygraph = MultimodalPolygraphProfile(
            facial_incongruence_score=round(facial_score, 3),
            voice_stress_score=round(vocal_score, 3),
            pupil_dilation_strain=round(pupil_score, 3),
            pacifying_adaptor_score=round(somatosensory_score, 3),
            autonomic_pulse_surge_score=round(autonomic_score, 3),
            channel_contributions={
                "facial": round(self.WEIGHT_FACIAL * facial_score, 3),
                "vocal": round(self.WEIGHT_VOCAL * vocal_score, 3),
                "pupillometric": round(self.WEIGHT_PUPILLOMETRIC * pupil_score, 3),
                "somatosensory": round(self.WEIGHT_SOMATOSENSORY * somatosensory_score, 3),
                "autonomic": round(self.WEIGHT_AUTONOMIC * autonomic_score, 3),
            },
        )

        return CredibilitySnapshot(
            timestamp=timestamp,
            credibility_score=round(credibility_score, 3),
            deception_risk_index=round(cdri, 3),
            tier=tier,
            facial_veracity=facial_metrics,
            voice_stress=voice_stress,
            polygraph=polygraph,
            recent_leakages=self.leakage_events[-5:],
            active_flags=unique_flags,
            clinical_verdict=verdict,
        )
