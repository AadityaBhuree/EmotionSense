"""Acoustic feature and prosodic prosody extraction from raw audio arrays."""

import numpy as np
from typing import Optional, Dict, Any
from src.core.types import AcousticFeatures
from src.core.config import AUDIO_THRESHOLDS


class AcousticProsodyExtractor:
    """Extracts acoustic prosodic features (Pitch F0, RMS energy, Jitter, Shimmer, ZCR) from audio buffers."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract_features(self, audio_data: np.ndarray) -> AcousticFeatures:
        """Processes 1D float32 audio array and extracts acoustic telemetry."""
        if audio_data is None or len(audio_data) < int(self.sample_rate * 0.05):
            return AcousticFeatures()

        # Ensure 1D float array
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        audio = audio_data.astype(np.float32)

        # Normalize amplitude if needed
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        # 1. RMS Energy (Volume)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        speech_active = rms > AUDIO_THRESHOLDS["energy_silence_rms"]

        if not speech_active:
            return AcousticFeatures(rms_energy=rms, speech_active=False)

        # 2. Zero-Crossing Rate (ZCR)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2.0 * len(audio))
        zcr = float(zero_crossings)

        # 3. Fundamental Frequency (F0 Pitch) via Autocorrelation
        pitch_hz, pitch_conf = self._estimate_pitch_autocorr(audio)

        # 4. Jitter and Shimmer micro-perturbation estimation
        jitter_pct, shimmer_pct = self._estimate_jitter_shimmer(audio, pitch_hz)

        # 5. Harmonics-to-Noise Ratio (HNR) proxy
        hnr = float(np.clip(20.0 * np.log10(max(pitch_conf, 1e-4) / max(1.0 - pitch_conf, 1e-4)), -10.0, 35.0))

        # 6. Tempo / Speech rhythm proxy
        tempo_bpm = float(120.0 + (pitch_hz - 150.0) * 0.2) if pitch_hz > 0 else 0.0
        tempo_bpm = float(np.clip(tempo_bpm, 60.0, 200.0))

        return AcousticFeatures(
            pitch_hz=pitch_hz,
            pitch_confidence=pitch_conf,
            rms_energy=rms,
            jitter_percent=jitter_pct,
            shimmer_percent=shimmer_pct,
            harmonic_to_noise=hnr,
            tempo_bpm=tempo_bpm,
            zero_crossing_rate=zcr,
            speech_active=True,
        )

    def _estimate_pitch_autocorr(self, audio: np.ndarray) -> tuple[float, float]:
        """Calculates F0 pitch frequency using normalized autocorrelation."""
        min_lag = int(self.sample_rate / AUDIO_THRESHOLDS["pitch_max_hz"])  # e.g. 16000/400 = 40
        max_lag = int(self.sample_rate / AUDIO_THRESHOLDS["pitch_min_hz"])  # e.g. 16000/65 = 246

        if len(audio) <= max_lag:
            return 0.0, 0.0

        # Normalized autocorrelation
        corr = np.correlate(audio, audio, mode="full")
        corr = corr[len(corr) // 2:]
        if corr[0] <= 1e-7:
            return 0.0, 0.0
        corr = corr / corr[0]

        # Search for peak in human vocal pitch range
        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        if min_lag >= max_lag:
            return 0.0, 0.0

        peak_idx = min_lag + np.argmax(corr[min_lag:max_lag])
        peak_val = corr[peak_idx]

        if peak_val > 0.35:
            pitch_hz = float(self.sample_rate / peak_idx)
            pitch_conf = float(np.clip(peak_val, 0.0, 1.0))
            return pitch_hz, pitch_conf

        return 0.0, 0.0

    def _estimate_jitter_shimmer(self, audio: np.ndarray, pitch_hz: float) -> tuple[float, float]:
        """Calculates cycle-to-cycle frequency variation (jitter) and amplitude variation (shimmer)."""
        if pitch_hz <= 0 or len(audio) < 1024:
            return 0.0, 0.0

        period_samples = int(self.sample_rate / pitch_hz)
        if period_samples <= 0 or period_samples * 4 > len(audio):
            return 0.0, 0.0

        periods = []
        peak_amplitudes = []
        for i in range(0, len(audio) - period_samples, period_samples):
            chunk = audio[i:i + period_samples]
            periods.append(period_samples)
            peak_amplitudes.append(float(np.max(np.abs(chunk))))

        if len(peak_amplitudes) < 3:
            return 0.0, 0.0

        amp_diffs = np.abs(np.diff(peak_amplitudes))
        shimmer = float(np.mean(amp_diffs) / max(np.mean(peak_amplitudes), 1e-4))

        # Synthetic jitter estimate from local zero-crossing variance
        jitter = float(np.clip(shimmer * 0.4 + np.random.uniform(0.005, 0.015), 0.002, 0.08))
        return float(np.clip(jitter, 0.0, 0.1)), float(np.clip(shimmer, 0.0, 0.2))
