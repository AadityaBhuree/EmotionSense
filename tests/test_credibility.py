"""Unit tests for Phase 13 Forensic Veracity Assessment & Affective Deception Leakage Engine."""

import time
import numpy as np
import pytest

from src.core.credibility_models import (
    VeracityTier,
    DeceptionFlag,
    MicroLeakageEvent,
    VoiceStressProfile,
    FacialVeracityMetrics,
    MultimodalPolygraphProfile,
    CredibilitySnapshot,
)
from src.analytics.credibility import CredibilityEngine


class TestCredibilityModels:
    """Verifies domain models, dataclass serialization, and enumeration contracts."""

    def test_default_models_serialization(self):
        leakage = MicroLeakageEvent()
        assert leakage.duration_ms == 120.0
        assert "duration_ms" in leakage.to_dict()

        voice_stress = VoiceStressProfile()
        assert not voice_stress.is_voice_stressed
        assert "cpp_db" in voice_stress.to_dict()

        facial_veracity = FacialVeracityMetrics()
        assert facial_veracity.duchenne_congruence == 0.85
        assert "duchenne_congruence" in facial_veracity.to_dict()

        polygraph = MultimodalPolygraphProfile()
        assert "channel_contributions" in polygraph.to_dict()

        snapshot = CredibilitySnapshot()
        d = snapshot.to_dict()
        assert "credibility_score" in d
        assert "facial_veracity" in d
        assert "voice_stress" in d
        assert "polygraph" in d


class TestCredibilityEngine:
    """Tests the forensic veracity, micro-leakage, and multimodal polygraph fusion logic."""

    @pytest.fixture
    def engine(self):
        return CredibilityEngine(
            leakage_flash_max_ms=250.0,
            duchenne_threshold=0.30,
            voice_stress_threshold=0.45,
            buffer_window_sec=10.0,
        )

    def test_duchenne_authentic_smile(self, engine):
        """Authentic smile with balanced AU12 (lip corner) and AU06 (cheek raiser)."""
        au = {"AU12": 0.80, "AU06": 0.75}
        metrics, leakages, flags = engine.compute_facial_veracity(au, macro_emotion="Joy")
        assert metrics.duchenne_congruence > 0.80
        assert metrics.duchenne_incongruence_index < 0.20
        assert DeceptionFlag.NON_DUCHENNE_MASKING.value not in flags

    def test_duchenne_masked_social_smile(self, engine):
        """Forced social smile with strong AU12 but absent AU06."""
        au = {"AU12": 0.85, "AU06": 0.02}
        metrics, leakages, flags = engine.compute_facial_veracity(au, macro_emotion="Joy")
        assert metrics.duchenne_incongruence_index > 0.30
        assert DeceptionFlag.NON_DUCHENNE_MASKING.value in flags

    def test_sneer_asymmetry_detection(self, engine):
        """Asymmetric sneer/smirk with prominent unilateral AU14."""
        au = {"AU14_L": 0.70, "AU14_R": 0.05}
        metrics, leakages, flags = engine.compute_facial_veracity(au, macro_emotion="Neutral")
        assert metrics.sneer_asymmetry_index > 0.40
        assert DeceptionFlag.FACIAL_ASYMMETRY_SNEER.value in flags

    def test_micro_momentary_leakage_detection(self, engine):
        """Detects a 100ms flash of negative fear/contempt breaking through a neutral mask."""
        now = time.time()
        # Feed baseline neutral frames
        for i in range(5):
            engine.compute_facial_veracity(
                {"AU12": 0.0, "AU04": 0.05},
                macro_emotion="Neutral",
                timestamp=now + (i * 0.033),
            )

        # Micro-flash of fear (AU20 + AU04) lasting ~100ms (3 frames)
        for i in range(5, 8):
            metrics, leakages, flags = engine.compute_facial_veracity(
                {"AU04": 0.65, "AU20": 0.60, "AU15": 0.30},
                macro_emotion="Joy",
                timestamp=now + (i * 0.033),
            )

        assert metrics.leakage_events_count >= 1
        assert DeceptionFlag.MICRO_FLASH_LEAKAGE.value in flags

    def test_voice_stress_calm_speech(self, engine):
        """Calm acoustic profile with low micro-tremor and healthy CPP."""
        synthetic_f0 = [140.0 + 0.5 * np.sin(2 * np.pi * 2 * t) for t in np.linspace(0, 1, 30)]
        profile, flags = engine.analyze_voice_stress(
            f0_trajectory=synthetic_f0,
            response_latency_sec=0.40,
        )
        assert profile.stress_index < 0.40
        assert not profile.is_voice_stressed
        assert DeceptionFlag.ACOUSTIC_MICRO_TREMOR.value not in flags

    def test_voice_stress_tremor_and_latency(self, engine):
        """Stressed speech with 10 Hz laryngeal micro-tremor and extended response latency."""
        # 10 Hz micro-tremor injected at 30 fps
        synthetic_f0 = [160.0 + 8.0 * np.sin(2 * np.pi * 10 * t) for t in np.linspace(0, 1, 30)]
        profile, flags = engine.analyze_voice_stress(
            f0_trajectory=synthetic_f0,
            response_latency_sec=3.2,
        )
        assert profile.micro_tremor_energy > 0.05
        assert DeceptionFlag.LATENCY_ELONGATION.value in flags

    def test_cpp_calculation_from_audio(self, engine):
        """Tests Cepstral Peak Prominence calculation from raw waveform."""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        # Harmonic signal (f0 = 150 Hz)
        audio = 0.5 * np.sin(2 * np.pi * 150 * t) + 0.25 * np.sin(2 * np.pi * 300 * t)
        cpp, slope = engine._compute_cpp_and_slope(audio, sr)
        assert cpp > 5.0
        assert slope < 0.0

    def test_multimodal_late_fusion_truthful(self, engine):
        """Organic truthful subject state produces high credibility and Veridical tier."""
        facial = FacialVeracityMetrics(duchenne_congruence=0.92, duchenne_incongruence_index=0.08)
        vocal = VoiceStressProfile(stress_index=0.15, is_voice_stressed=False)

        snapshot = engine.fuse_credibility_assessment(
            facial_metrics=facial,
            voice_stress=vocal,
            pupil_cpr=1.02,
            pacifying_adaptor_active=False,
            pulse_bpm=74.0,
        )
        assert snapshot.credibility_score > 0.75
        assert snapshot.deception_risk_index < 0.25
        assert snapshot.tier == VeracityTier.VERIDICAL_AUTHENTIC.value

    def test_multimodal_late_fusion_high_deception_risk(self, engine):
        """Concurrent non-Duchenne smile, micro-leakage, voice stress, and pulse surge."""
        facial = FacialVeracityMetrics(
            duchenne_congruence=0.10,
            duchenne_incongruence_index=0.90,
            sneer_asymmetry_index=0.60,
            micro_leakage_detected=True,
        )
        vocal = VoiceStressProfile(stress_index=0.85, is_voice_stressed=True)

        snapshot = engine.fuse_credibility_assessment(
            facial_metrics=facial,
            voice_stress=vocal,
            pupil_cpr=1.45,
            pacifying_adaptor_active=True,
            pulse_bpm=108.0,
        )
        assert snapshot.deception_risk_index >= 0.70
        assert snapshot.tier in [
            VeracityTier.SUSPICIOUS_INCONGRUENCE.value,
            VeracityTier.HIGH_DECEPTION_RISK.value,
        ]
        assert DeceptionFlag.PACIFYING_ADAPTOR_SURGE.value in snapshot.active_flags
        assert DeceptionFlag.AUTONOMIC_PULSE_SURGE.value in snapshot.active_flags
