"""Tests for Phase 6 Cross-Modal Attentive Fusion (CMAF) Transformer and Congruence Engine."""

import numpy as np
import pytest
from src.core.types import (
    AcousticFeatures,
    AcousticSERResult,
    AffectVector,
    CrossModalFusionResult,
    FacialActionUnits,
    TextEmotionResult,
    VisionEmotionResult,
    VoiceEmotionResult,
)
from src.fusion.cross_modal_fusion import CrossModalAttentionFusion


@pytest.fixture
def cmaf():
    return CrossModalAttentionFusion()


@pytest.fixture
def sample_vision():
    return VisionEmotionResult(
        dominant_emotion="joy",
        confidence=0.88,
        probabilities={"joy": 0.82, "surprise": 0.10, "neutral": 0.05, "sadness": 0.01, "anger": 0.01, "fear": 0.01, "disgust": 0.0, "contempt": 0.0},
        action_units=FacialActionUnits(lip_corner_puller=0.85, cheek_raiser=0.75),
        affect=AffectVector(valence=0.78, arousal=0.52, dominance=0.45),
        face_detected=True,
    )


@pytest.fixture
def sample_audio_ser():
    return AcousticSERResult(
        emotion_scores={"joy": 0.75, "surprise": 0.15, "neutral": 0.05, "sadness": 0.02, "anger": 0.01, "fear": 0.01, "disgust": 0.01, "contempt": 0.0},
        dominant_emotion="joy",
        confidence=0.75,
        vad=AffectVector(valence=0.65, arousal=0.48, dominance=0.40),
        embedding=[0.2] * 64,
        backend="onnx_quantized",
    )


@pytest.fixture
def sample_text():
    return TextEmotionResult(
        text="We achieved outstanding results on our quarterly targets!",
        dominant_emotion="joy",
        confidence=0.92,
        probabilities={"joy": 0.90, "surprise": 0.05, "neutral": 0.03, "sadness": 0.01, "anger": 0.01, "fear": 0.0, "disgust": 0.0, "contempt": 0.0},
        affect=AffectVector(valence=0.85, arousal=0.60, dominance=0.55),
    )


def test_cmaf_initialization(cmaf):
    assert cmaf.embed_dim == 64
    assert cmaf.num_heads == 4
    assert len(cmaf.labels) == 8


def test_trimodal_cross_modal_fusion(cmaf, sample_vision, sample_audio_ser, sample_text):
    result = cmaf.fuse(vision=sample_vision, audio=sample_audio_ser, text=sample_text)

    assert isinstance(result, CrossModalFusionResult)
    assert result.dominant_emotion == "joy"
    assert 0.0 <= result.confidence <= 1.0
    assert np.isclose(sum(result.emotion_scores.values()), 1.0, atol=1e-3)

    # 3D VAD continuous coords
    assert -1.0 <= result.vad.valence <= 1.0
    assert -1.0 <= result.vad.arousal <= 1.0
    assert -1.0 <= result.vad.dominance <= 1.0

    # Cross-attention weights
    attn = result.attention_weights
    assert attn is not None
    assert len(attn.attention_matrix) == 3
    assert len(attn.attention_matrix[0]) == 3
    assert 0.0 <= attn.text_to_vision <= 1.0
    assert 0.0 <= attn.text_to_audio <= 1.0

    # Congruence
    assert result.congruence is not None
    assert result.congruence.congruence_tier in ["HARMONIOUS", "MODERATE_TENSION"]
    assert result.congruence.congruence_score >= 50.0

    # Modality contributions
    contribs = result.modality_contributions
    assert np.isclose(sum(contribs.values()), 1.0, atol=1e-2)
    assert contribs["vision"] > 0.0
    assert contribs["audio"] > 0.0
    assert contribs["text"] > 0.0


def test_missing_modalities_graceful_handling(cmaf, sample_vision):
    # Vision only (no audio, no text)
    res_vis_only = cmaf.fuse(vision=sample_vision, audio=None, text=None)
    assert res_vis_only.dominant_emotion == "joy"
    assert res_vis_only.attention_weights.modality_gates["vision"] > 0.5
    assert res_vis_only.attention_weights.modality_gates["audio"] == 0.0
    assert res_vis_only.attention_weights.modality_gates["text"] == 0.0

    # All None (offline / black frame)
    res_none = cmaf.fuse(vision=None, audio=None, text=None)
    assert res_none.dominant_emotion == "neutral"
    assert res_none.congruence.congruence_score == 100.0


def test_masked_affect_detection(cmaf):
    # Smiling face (joy AU12 = 0.8)
    smiling_face = VisionEmotionResult(
        dominant_emotion="joy",
        confidence=0.85,
        probabilities={"joy": 0.80, "neutral": 0.15, "anger": 0.05, "sadness": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "contempt": 0.0},
        action_units=FacialActionUnits(lip_corner_puller=0.82),
        affect=AffectVector(valence=0.75, arousal=0.30, dominance=0.20),
        face_detected=True,
    )

    # Hostile / Angry voice
    angry_voice = VoiceEmotionResult(
        dominant_emotion="anger",
        confidence=0.85,
        probabilities={"anger": 0.82, "disgust": 0.10, "neutral": 0.08, "joy": 0.0, "sadness": 0.0, "fear": 0.0, "surprise": 0.0, "contempt": 0.0},
        acoustics=AcousticFeatures(speech_active=True, rms_energy=0.15, jitter_percent=0.035),
        affect=AffectVector(valence=-0.65, arousal=0.70, dominance=0.40),
    )

    # Passive-aggressive text
    sarcastic_text = TextEmotionResult(
        text="Oh sure, absolutely fantastic job ruining everything.",
        dominant_emotion="anger",
        confidence=0.80,
        probabilities={"anger": 0.75, "contempt": 0.15, "neutral": 0.10, "joy": 0.0, "sadness": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0},
        affect=AffectVector(valence=-0.70, arousal=0.50, dominance=0.30),
    )

    result = cmaf.fuse(vision=smiling_face, audio=angry_voice, text=sarcastic_text)

    # Should detect masked affect / discord
    assert result.congruence.congruence_tier in ["MASKED_AFFECT", "SIGNIFICANT_DISCORD"]
    assert result.congruence.congruence_score <= 45.0
    assert result.congruence.primary_conflict is not None
    assert "mask" in result.congruence.primary_conflict.lower() or "smiling" in result.congruence.primary_conflict.lower()
