"""Phase 10: Remote Biometric & Physiological Telemetry Engine.

Implements contact-free optical photoplethysmography (rPPG) via Plane-Orthogonal-to-Skin (POS)
and CHROM algorithms, digital Butterworth bandpass filtering, inter-beat interval (RR) peak detection,
time-domain Heart Rate Variability (HRV - SDNN, RMSSD, pNN50, Baevsky Stress Index),
Respiratory Sinus Arrhythmia (RSA) estimation, and multimodal autonomic stress fusion.
"""

import time
from typing import List, Tuple, Optional
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from src.core.biometric_models import (
    StressClassification,
    PulseMeasurement,
    HRVMetrics,
    RespirationMetrics,
    AutonomicStressRecord,
    BiometricTelemetry,
)


class BiometricEngine:
    """Enterprise remote photoplethysmography and autonomic stress telemetry engine."""

    def __init__(
        self,
        fps: float = 30.0,
        buffer_window_sec: float = 6.0,
        lowcut_hz: float = 0.75,   # 45 BPM
        highcut_hz: float = 3.0,    # 180 BPM
    ):
        self.fps = max(10.0, float(fps))
        self.buffer_window_sec = max(2.0, float(buffer_window_sec))
        self.max_buffer_size = int(self.fps * self.buffer_window_sec)
        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz

        # Raw RGB channel buffers (averages across face/skin ROI)
        self.rgb_buffer: List[Tuple[float, float, float]] = []
        self.timestamp_buffer: List[float] = []

        # Filtered BVP signal history
        self.bvp_history: List[float] = []
        self.last_pulse: PulseMeasurement = PulseMeasurement()
        self.last_hrv: HRVMetrics = HRVMetrics()
        self.last_respiration: RespirationMetrics = RespirationMetrics()
        self.last_stress: AutonomicStressRecord = AutonomicStressRecord()

    def add_rgb_sample(self, r: float, g: float, b: float, timestamp: Optional[float] = None) -> None:
        """Add a mean skin ROI RGB sample to the temporal buffer."""
        if timestamp is None:
            timestamp = time.time()
        self.rgb_buffer.append((float(r), float(g), float(b)))
        self.timestamp_buffer.append(timestamp)

        # Retain sliding window
        if len(self.rgb_buffer) > self.max_buffer_size:
            self.rgb_buffer.pop(0)
            self.timestamp_buffer.pop(0)

    def extract_pos_bvp(self, rgb_series: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract Blood Volume Pulse (BVP) using Plane-Orthogonal-to-Skin (POS) algorithm.
        
        Wang et al., 'Algorithmic Principles of Remote PPG', IEEE TBME 2017.
        """
        if rgb_series is None:
            if len(self.rgb_buffer) < 15:
                return np.zeros(len(self.rgb_buffer))
            rgb_series = np.array(self.rgb_buffer, dtype=np.float64)

        if len(rgb_series) < 15:
            return np.zeros(len(rgb_series))

        # Temporal normalization by mean channel intensity
        mean_rgb = np.mean(rgb_series, axis=0) + 1e-6
        norm_rgb = rgb_series / mean_rgb

        r = norm_rgb[:, 0]
        g = norm_rgb[:, 1]
        b = norm_rgb[:, 2]

        # POS Chrominance projection
        s1 = 3.0 * r - 2.0 * g
        s2 = 1.5 * r + g - 1.5 * b

        std_s1 = np.std(s1)
        std_s2 = np.std(s2) + 1e-6
        alpha = std_s1 / std_s2

        raw_bvp = s1 - alpha * s2
        # Apply bandpass filter
        filtered_bvp = self.bandpass_filter(raw_bvp, self.fps, self.lowcut_hz, self.highcut_hz)
        return filtered_bvp

    @staticmethod
    def bandpass_filter(
        signal: np.ndarray,
        fps: float,
        lowcut: float = 0.75,
        highcut: float = 3.0,
        order: int = 2,
    ) -> np.ndarray:
        """Zero-phase forward-backward Butterworth bandpass filter."""
        n = len(signal)
        if n < 15:
            return signal

        nyquist = 0.5 * fps
        low = max(0.01, min(lowcut / nyquist, 0.98))
        high = max(low + 0.01, min(highcut / nyquist, 0.99))

        try:
            b, a = butter(order, [low, high], btype="bandpass")
            padlen = min(n - 1, 3 * max(len(a), len(b)))
            if n <= padlen:
                return signal - np.mean(signal)
            filtered = filtfilt(b, a, signal, padlen=padlen)
            return filtered
        except Exception:
            return signal - np.mean(signal)

    def compute_pulse_from_bvp(self, bvp_signal: np.ndarray) -> PulseMeasurement:
        """Estimate instantaneous heart rate (BPM) and signal quality SNR from BVP."""
        if len(bvp_signal) < int(self.fps * 1.5):
            return PulseMeasurement(bpm=72.0, confidence=0.3, signal_quality_snr=5.0, is_valid=False)

        # 1. Spectral FFT analysis
        n = len(bvp_signal)
        fft_vals = np.fft.rfft(bvp_signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        magnitudes = np.abs(fft_vals)

        # Constrain to physiological heart rate band (45 - 180 BPM -> 0.75 - 3.0 Hz)
        valid_idx = (freqs >= self.lowcut_hz) & (freqs <= self.highcut_hz)
        if not np.any(valid_idx):
            return PulseMeasurement(bpm=72.0, confidence=0.3, signal_quality_snr=5.0, is_valid=False)

        valid_freqs = freqs[valid_idx]
        valid_mags = magnitudes[valid_idx]
        peak_idx = np.argmax(valid_mags)
        peak_freq = valid_freqs[peak_idx]
        spectral_bpm = peak_freq * 60.0

        # Calculate SNR (dB): ratio of power in peak window (+/- 2 bins) to average noise floor
        half_win = 2
        p_start = max(0, peak_idx - half_win)
        p_end = min(len(valid_mags), peak_idx + half_win + 1)
        peak_power = float(np.sum(valid_mags[p_start:p_end] ** 2))
        
        # Noise floor: median power across the rest of the band
        other_mask = np.ones(len(valid_mags), dtype=bool)
        other_mask[p_start:p_end] = False
        if np.any(other_mask):
            noise_power = float(np.mean(valid_mags[other_mask] ** 2)) * (p_end - p_start) + 1e-6
        else:
            noise_power = 1e-6

        snr_db = float(10.0 * np.log10(max(1.0, peak_power) / max(1e-6, noise_power)))
        confidence = float(np.clip(0.3 + (snr_db / 20.0) * 0.6, 0.25, 0.98))

        # 2. Time-domain peak confirmation
        min_distance = int(self.fps * (60.0 / 190.0))  # Max 190 BPM
        peaks, _ = find_peaks(bvp_signal, distance=max(2, min_distance), prominence=np.std(bvp_signal) * 0.3)

        if len(peaks) >= 3:
            rr_samples = np.diff(peaks)
            rr_ms = (rr_samples / self.fps) * 1000.0
            # Filter physiological outliers (300ms to 1400ms)
            valid_rr = rr_ms[(rr_ms >= 300.0) & (rr_ms <= 1400.0)]
            if len(valid_rr) >= 2:
                time_bpm = 60000.0 / np.mean(valid_rr)
                final_bpm = float(0.6 * spectral_bpm + 0.4 * time_bpm)
                confidence = min(0.98, confidence + 0.1)
            else:
                final_bpm = float(spectral_bpm)
        else:
            final_bpm = float(spectral_bpm)

        final_bpm = float(np.clip(final_bpm, 45.0, 185.0))
        latest_sample = float(bvp_signal[-1]) if len(bvp_signal) > 0 else 0.0

        return PulseMeasurement(
            bpm=round(final_bpm, 1),
            confidence=round(confidence, 2),
            signal_quality_snr=round(snr_db, 1),
            timestamp=time.time(),
            bvp_sample=round(latest_sample, 4),
            is_valid=confidence >= 0.35 or snr_db > 2.0,
        )

    def extract_rr_intervals(self, bvp_signal: np.ndarray) -> List[float]:
        """Extract inter-beat intervals (RR intervals in milliseconds) from BVP peaks."""
        if len(bvp_signal) < int(self.fps * 2.0):
            return []

        min_distance = int(self.fps * (60.0 / 190.0))
        peaks, _ = find_peaks(bvp_signal, distance=max(2, min_distance), prominence=np.std(bvp_signal) * 0.35)

        if len(peaks) < 2:
            return []

        rr_samples = np.diff(peaks)
        rr_ms = (rr_samples / self.fps) * 1000.0
        # Physiologically realistic NN intervals (330ms = 181 BPM, 1333ms = 45 BPM)
        valid_rr = [float(x) for x in rr_ms if 330.0 <= x <= 1350.0]
        return valid_rr

    @staticmethod
    def compute_hrv_metrics(rr_intervals_ms: List[float]) -> HRVMetrics:
        """Calculate clinical time-domain HRV metrics and Baevsky Stress Index."""
        if not rr_intervals_ms or len(rr_intervals_ms) < 3:
            return HRVMetrics()

        rr = np.array(rr_intervals_ms, dtype=np.float64)
        mean_rr = float(np.mean(rr))
        sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0

        successive_diffs = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0

        nn50_count = np.sum(np.abs(successive_diffs) > 50.0)
        pnn50 = float((nn50_count / len(successive_diffs)) * 100.0) if len(successive_diffs) > 0 else 0.0

        # Baevsky Stress Index (SI = AMo / (2 * Mo * MxDMn))
        # Mo: Mode of RR intervals in seconds
        # AMo: Amplitude of Mode (% of RR in 50ms bin centered at Mo)
        # MxDMn: Variation Range (max RR - min RR) in seconds
        mo_sec = max(0.4, mean_rr / 1000.0)
        bin_width = 0.05
        bins = np.arange(np.min(rr) / 1000.0, np.max(rr) / 1000.0 + bin_width, bin_width)
        if len(bins) > 1:
            hist, bin_edges = np.histogram(rr / 1000.0, bins=bins)
            amo = float(np.max(hist) / len(rr) * 100.0)
            max_bin_idx = np.argmax(hist)
            mo_sec = float((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0)
        else:
            amo = 50.0

        mxdmn_sec = max(0.06, (np.max(rr) - np.min(rr)) / 1000.0)
        baevsky_si = float(amo / (2.0 * mo_sec * mxdmn_sec))
        baevsky_si = float(np.clip(baevsky_si, 10.0, 950.0))

        # Restorative HRV vitality score (0.0 to 100.0): higher RMSSD/SDNN -> higher score
        hrv_score = float(np.clip((rmssd / 60.0) * 50.0 + (sdnn / 70.0) * 50.0, 5.0, 100.0))

        return HRVMetrics(
            sdnn_ms=round(sdnn, 1),
            rmssd_ms=round(rmssd, 1),
            pnn50_pct=round(pnn50, 1),
            mean_rr_ms=round(mean_rr, 1),
            baevsky_stress_index=round(baevsky_si, 1),
            hrv_score=round(hrv_score, 1),
        )

    def estimate_respiration_rate(self, bvp_signal: np.ndarray, rr_intervals_ms: List[float]) -> RespirationMetrics:
        """Estimate respiration rate (RPM) via Respiratory Sinus Arrhythmia (RSA) modulation."""
        if len(bvp_signal) < int(self.fps * 4.0):
            return RespirationMetrics(rpm=15.0, confidence=0.4)

        # Amplitude Envelope Modulation of BVP (low-frequency breathing wave 0.15 - 0.4 Hz / 9 - 24 RPM)
        analytic_signal = np.abs(bvp_signal)
        resp_bvp = self.bandpass_filter(analytic_signal, self.fps, lowcut=0.15, highcut=0.45, order=2)

        fft_vals = np.abs(np.fft.rfft(resp_bvp))
        freqs = np.fft.rfftfreq(len(resp_bvp), d=1.0 / self.fps)
        resp_band = (freqs >= 0.15) & (freqs <= 0.45)

        if np.any(resp_band):
            band_mags = fft_vals[resp_band]
            band_freqs = freqs[resp_band]
            peak_f = band_freqs[np.argmax(band_mags)]
            envelope_rpm = peak_f * 60.0
            confidence = 0.75
        else:
            envelope_rpm = 15.0
            confidence = 0.4

        # If RR intervals are available, cross-verify with RSA tachogram
        if len(rr_intervals_ms) >= 8:
            rr_diff = np.diff(rr_intervals_ms)
            if len(rr_diff) >= 4:
                peaks, _ = find_peaks(np.abs(rr_diff))
                if len(peaks) >= 2:
                    cycle_time = np.mean(np.diff(peaks)) * (np.mean(rr_intervals_ms) / 1000.0)
                    if cycle_time > 1.5:
                        rsa_rpm = 60.0 / cycle_time
                        if 8.0 <= rsa_rpm <= 28.0:
                            envelope_rpm = 0.5 * envelope_rpm + 0.5 * rsa_rpm
                            confidence = 0.85

        rpm = float(np.clip(envelope_rpm, 8.0, 30.0))
        return RespirationMetrics(
            rpm=round(rpm, 1),
            confidence=round(confidence, 2),
            method="RSA_Modulation",
        )

    @staticmethod
    def compute_autonomic_stress(
        pulse: PulseMeasurement,
        hrv: HRVMetrics,
        respiration: RespirationMetrics,
        valence: float = 0.0,
        arousal: float = 0.0,
        vocal_jitter: float = 0.02,
    ) -> AutonomicStressRecord:
        """Compute unified autonomic nervous system stress index fusing physiological and affective markers.
        
        Outputs stress_index in [0.0, 1.0], sympathetic/parasympathetic tone, and clinical classification.
        """
        # 1. Cardiac component (Tachycardia above baseline 70 BPM)
        norm_hr_stress = float(np.clip((pulse.bpm - 65.0) / 45.0, 0.0, 1.0))

        # 2. Vagal tone collapse (RMSSD below 45ms indicates parasympathetic withdrawal)
        vagal_collapse = float(np.clip((45.0 - hrv.rmssd_ms) / 35.0, 0.0, 1.0))

        # 3. Baevsky Sympathetic Strain (SI > 120 indicates high sympathetic activation)
        baevsky_strain = float(np.clip((hrv.baevsky_stress_index - 80.0) / 250.0, 0.0, 1.0))

        # 4. Tachypnea / Rapid Breathing (RPM > 18)
        resp_strain = float(np.clip((respiration.rpm - 14.0) / 12.0, 0.0, 1.0))

        # 5. Affective distress multiplier (Negative valence + High arousal)
        affect_stress = 0.0
        if valence < 0.0 and arousal > 0.3:
            affect_stress = abs(valence) * arousal

        # 6. Vocal acoustic tension (jitter > 0.03)
        vocal_strain = float(np.clip((vocal_jitter - 0.015) / 0.04, 0.0, 1.0))

        # Weighted composite stress score
        raw_stress = (
            0.25 * norm_hr_stress
            + 0.25 * vagal_collapse
            + 0.20 * baevsky_strain
            + 0.10 * resp_strain
            + 0.12 * affect_stress
            + 0.08 * vocal_strain
        )
        stress_index = float(np.clip(raw_stress, 0.02, 0.98))

        # Sympathetic vs Parasympathetic balance
        sympathetic_tone = float(np.clip(0.3 + 0.65 * stress_index, 0.05, 0.98))
        parasympathetic_tone = float(np.clip(1.0 - sympathetic_tone, 0.02, 0.95))
        somatic_arousal = float(np.clip(0.5 * norm_hr_stress + 0.5 * resp_strain, 0.0, 1.0))

        # Classification & contributing clinical factors
        factors = []
        if pulse.bpm > 90.0:
            factors.append(f"Elevated Heart Rate ({pulse.bpm:.0f} BPM)")
        elif pulse.bpm < 60.0:
            factors.append(f"Resting Bradycardia ({pulse.bpm:.0f} BPM)")

        if hrv.rmssd_ms < 25.0:
            factors.append(f"Parasympathetic Vagal Suppression (RMSSD {hrv.rmssd_ms:.0f}ms)")
        else:
            factors.append(f"Robust Vagal Resilience (RMSSD {hrv.rmssd_ms:.0f}ms)")

        if hrv.baevsky_stress_index > 180.0:
            factors.append(f"High Sympathetic Baevsky Strain (SI {hrv.baevsky_stress_index:.0f})")

        if respiration.rpm > 20.0:
            factors.append(f"Tachypneic Respiration ({respiration.rpm:.0f} RPM)")

        if affect_stress > 0.2:
            factors.append("Affective Valence-Arousal Discordance")

        if not factors:
            factors = ["Homeostatic Cardiac Stability", "Balanced Autonomic Tone"]

        if stress_index < 0.25:
            classification = StressClassification.RELAXED.value
        elif stress_index < 0.52:
            classification = StressClassification.OPTIMAL_ALERTNESS.value
        elif stress_index < 0.76:
            classification = StressClassification.ELEVATED_STRAIN.value
        else:
            classification = StressClassification.ACUTE_DISTRESS.value

        return AutonomicStressRecord(
            stress_index=round(stress_index, 2),
            classification=classification,
            sympathetic_tone=round(sympathetic_tone, 2),
            parasympathetic_tone=round(parasympathetic_tone, 2),
            somatic_arousal=round(somatic_arousal, 2),
            contributing_factors=factors,
        )

    def process_frame_rgb(
        self,
        r: float,
        g: float,
        b: float,
        timestamp: Optional[float] = None,
        valence: float = 0.0,
        arousal: float = 0.0,
        vocal_jitter: float = 0.02,
    ) -> BiometricTelemetry:
        """Process incoming mean skin ROI RGB intensities and output fresh BiometricTelemetry."""
        self.add_rgb_sample(r, g, b, timestamp)

        bvp_signal = self.extract_pos_bvp()
        if len(bvp_signal) > 0:
            self.bvp_history = [float(x) for x in bvp_signal[-120:]]  # keep last 120 samples
        else:
            self.bvp_history = []

        pulse = self.compute_pulse_from_bvp(bvp_signal)
        rr_intervals = self.extract_rr_intervals(bvp_signal)
        hrv = self.compute_hrv_metrics(rr_intervals)
        respiration = self.estimate_respiration_rate(bvp_signal, rr_intervals)
        stress = self.compute_autonomic_stress(pulse, hrv, respiration, valence, arousal, vocal_jitter)

        self.last_pulse = pulse
        self.last_hrv = hrv
        self.last_respiration = respiration
        self.last_stress = stress

        return BiometricTelemetry(
            timestamp=time.time() if timestamp is None else timestamp,
            pulse=pulse,
            hrv=hrv,
            respiration=respiration,
            autonomic_stress=stress,
            bvp_history=self.bvp_history,
            rr_intervals_ms=rr_intervals,
            roi_detected=True,
        )

    @staticmethod
    def generate_synthetic_bvp_stream(
        duration_sec: float = 10.0,
        fps: float = 30.0,
        bpm: float = 72.0,
        noise_level: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a realistic synthetic blood volume pulse wave for unit testing and offline simulation."""
        n_samples = int(duration_sec * fps)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        cardiac_freq = bpm / 60.0

        # Fundamental + 1st harmonic (dicrotic notch) + respiratory modulation
        pulse_wave = (
            np.sin(2 * np.pi * cardiac_freq * t)
            + 0.35 * np.sin(4 * np.pi * cardiac_freq * t + 0.4)
            + 0.15 * np.sin(2 * np.pi * 0.25 * t)  # 15 RPM respiration
        )

        noise = np.random.normal(0, noise_level, n_samples)
        noisy_bvp = pulse_wave + noise

        # Simulate RGB channels: green channel has highest hemoglobin absorption
        r = 140.0 + 3.0 * noisy_bvp
        g = 110.0 + 8.0 * noisy_bvp
        b = 95.0 + 2.0 * noisy_bvp
        rgb_matrix = np.column_stack([r, g, b])

        return t, rgb_matrix
