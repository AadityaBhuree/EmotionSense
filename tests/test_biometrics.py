"""Comprehensive Unit and Integration Tests for Phase 10 Biometrics & rPPG Engine."""

import pytest
import numpy as np

from src.core.biometric_models import (
    StressClassification,
    PulseMeasurement,
    HRVMetrics,
    RespirationMetrics,
    AutonomicStressRecord,
    BiometricTelemetry,
)
from src.analytics.biometrics import BiometricEngine


class TestBiometricModels:
    """Test domain data contracts and serializations."""

    def test_pulse_measurement_to_dict(self):
        pulse = PulseMeasurement(bpm=74.5, confidence=0.91, signal_quality_snr=14.2)
        d = pulse.to_dict()
        assert d["bpm"] == 74.5
        assert d["confidence"] == 0.91
        assert d["signal_quality_snr"] == 14.2
        assert d["is_valid"] is True

    def test_hrv_metrics_to_dict(self):
        hrv = HRVMetrics(sdnn_ms=52.0, rmssd_ms=45.0, baevsky_stress_index=90.0)
        d = hrv.to_dict()
        assert d["sdnn_ms"] == 52.0
        assert d["rmssd_ms"] == 45.0
        assert d["baevsky_stress_index"] == 90.0
        assert d["hrv_score"] == 75.0

    def test_autonomic_stress_record_to_dict(self):
        stress = AutonomicStressRecord(
            stress_index=0.65,
            classification=StressClassification.ELEVATED_STRAIN.value,
            sympathetic_tone=0.72,
            parasympathetic_tone=0.28,
        )
        d = stress.to_dict()
        assert d["stress_index"] == 0.65
        assert d["classification"] == "Elevated_Strain"
        assert d["sympathetic_tone"] == 0.72

    def test_biometric_telemetry_complete_serialization(self):
        telem = BiometricTelemetry(
            timestamp=1700000000.0,
            pulse=PulseMeasurement(bpm=80.0),
            hrv=HRVMetrics(rmssd_ms=35.0),
            respiration=RespirationMetrics(rpm=16.0),
            autonomic_stress=AutonomicStressRecord(stress_index=0.40),
            bvp_history=[0.1, 0.2, -0.1],
            rr_intervals_ms=[750.0, 740.0, 760.0],
        )
        d = telem.to_dict()
        assert d["timestamp"] == 1700000000.0
        assert d["pulse"]["bpm"] == 80.0
        assert d["hrv"]["rmssd_ms"] == 35.0
        assert d["respiration"]["rpm"] == 16.0
        assert d["autonomic_stress"]["stress_index"] == 0.40
        assert len(d["bvp_history"]) == 3


class TestBiometricEngine:
    """Test signal processing, POS algorithm, and physiological derivations."""

    @pytest.fixture
    def engine(self):
        return BiometricEngine(fps=30.0, buffer_window_sec=6.0)

    def test_engine_initialization(self, engine):
        assert engine.fps == 30.0
        assert engine.max_buffer_size == 180
        assert len(engine.rgb_buffer) == 0

    def test_synthetic_bvp_stream_generation(self):
        t, rgb_matrix = BiometricEngine.generate_synthetic_bvp_stream(
            duration_sec=8.0, fps=30.0, bpm=75.0, noise_level=0.01
        )
        assert len(t) == 240
        assert rgb_matrix.shape == (240, 3)
        # Green channel has strongest amplitude
        assert np.mean(rgb_matrix[:, 1]) > 100.0

    def test_pos_algorithm_extraction(self, engine):
        _, rgb_matrix = BiometricEngine.generate_synthetic_bvp_stream(
            duration_sec=8.0, fps=30.0, bpm=72.0, noise_level=0.02
        )
        bvp = engine.extract_pos_bvp(rgb_matrix)
        assert len(bvp) == 240
        assert np.std(bvp) > 0.001
        assert not np.isnan(bvp).any()

    def test_pulse_rate_estimation_accuracy(self, engine):
        target_bpm = 75.0
        _, rgb_matrix = BiometricEngine.generate_synthetic_bvp_stream(
            duration_sec=10.0, fps=30.0, bpm=target_bpm, noise_level=0.02
        )
        bvp = engine.extract_pos_bvp(rgb_matrix)
        pulse = engine.compute_pulse_from_bvp(bvp)

        assert pulse.is_valid is True
        assert abs(pulse.bpm - target_bpm) <= 5.0
        assert pulse.signal_quality_snr > 3.0
        assert pulse.confidence >= 0.5

    def test_hrv_metrics_calculation(self):
        # 10 intervals with mean ~810ms (74 BPM) with healthy resting physiological variance
        intervals = [760.0, 830.0, 790.0, 870.0, 800.0, 840.0, 780.0, 890.0, 820.0, 850.0]
        hrv = BiometricEngine.compute_hrv_metrics(intervals)

        assert 800.0 <= hrv.mean_rr_ms <= 840.0
        assert hrv.sdnn_ms > 15.0
        assert hrv.rmssd_ms > 15.0
        assert 10.0 <= hrv.baevsky_stress_index <= 500.0
        assert 10.0 <= hrv.hrv_score <= 100.0

    def test_respiration_rate_estimation(self, engine):
        _, rgb_matrix = BiometricEngine.generate_synthetic_bvp_stream(
            duration_sec=10.0, fps=30.0, bpm=70.0, noise_level=0.02
        )
        bvp = engine.extract_pos_bvp(rgb_matrix)
        rr_intervals = [850.0, 840.0, 860.0, 870.0, 840.0, 830.0, 850.0, 865.0]
        resp = engine.estimate_respiration_rate(bvp, rr_intervals)

        assert 10.0 <= resp.rpm <= 25.0
        assert resp.confidence > 0.3
        assert resp.method == "RSA_Modulation"

    def test_autonomic_stress_computation_relaxed_vs_acute(self):
        # Relaxed scenario: low HR, high RMSSD, low Baevsky SI, positive valence
        relaxed_pulse = PulseMeasurement(bpm=62.0)
        relaxed_hrv = HRVMetrics(rmssd_ms=65.0, baevsky_stress_index=45.0)
        relaxed_resp = RespirationMetrics(rpm=13.0)
        stress_relaxed = BiometricEngine.compute_autonomic_stress(
            relaxed_pulse, relaxed_hrv, relaxed_resp, valence=0.6, arousal=0.2
        )
        assert stress_relaxed.classification in ["Relaxed", "Optimal_Alertness"]
        assert stress_relaxed.stress_index < 0.40
        assert stress_relaxed.parasympathetic_tone > stress_relaxed.sympathetic_tone

        # Acute distress scenario: tachycardia 115 BPM, low RMSSD 14ms, high Baevsky 320, negative valence, high arousal
        stressed_pulse = PulseMeasurement(bpm=115.0)
        stressed_hrv = HRVMetrics(rmssd_ms=14.0, baevsky_stress_index=320.0)
        stressed_resp = RespirationMetrics(rpm=24.0)
        stress_acute = BiometricEngine.compute_autonomic_stress(
            stressed_pulse, stressed_hrv, stressed_resp, valence=-0.75, arousal=0.85, vocal_jitter=0.05
        )
        assert stress_acute.classification in ["Elevated_Strain", "Acute_Distress"]
        assert stress_acute.stress_index > 0.60
        assert stress_acute.sympathetic_tone > stress_acute.parasympathetic_tone
        assert any("Elevated Heart Rate" in factor or "Vagal" in factor for factor in stress_acute.contributing_factors)

    def test_full_pipeline_streaming(self, engine):
        _, rgb_matrix = BiometricEngine.generate_synthetic_bvp_stream(
            duration_sec=6.0, fps=30.0, bpm=72.0, noise_level=0.02
        )
        telemetry = None
        for i in range(len(rgb_matrix)):
            r, g, b = rgb_matrix[i]
            telemetry = engine.process_frame_rgb(r, g, b, valence=0.1, arousal=0.3)

        assert telemetry is not None
        assert isinstance(telemetry, BiometricTelemetry)
        assert len(telemetry.bvp_history) > 0
        assert 50.0 <= telemetry.pulse.bpm <= 100.0
