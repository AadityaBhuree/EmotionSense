"""Tests for Phase 6 Deep Speech Emotion Recognition (SER) Classifier."""

import numpy as np
import pytest
from src.audio.deep_ser import DeepSpeechEmotionClassifier
from src.core.types import AcousticSERResult


@pytest.fixture
def ser_classifier():
    return DeepSpeechEmotionClassifier(backend="onnx_quantized")


@pytest.fixture
def synthetic_speech():
    """Generates synthetic voiced audio signal."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 200 Hz fundamental pitch + harmonics + noise
    y = 0.4 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 400 * t) + 0.05 * np.random.randn(sr)
    return y.astype(np.float32), sr


def test_ser_initialization():
    classifier = DeepSpeechEmotionClassifier()
    assert classifier.sample_rate == 16000
    assert classifier.embed_dim == 64
    assert classifier.backend in ["deep_neural", "onnx_quantized", "prosody_heuristic"]


def test_classify_silence(ser_classifier):
    sr = 16000
    silent_audio = np.zeros(sr, dtype=np.float32)
    res = ser_classifier.classify_waveform(silent_audio, sr)

    assert isinstance(res, AcousticSERResult)
    assert res.dominant_emotion == "neutral"
    assert res.confidence >= 0.8
    assert np.isclose(sum(res.emotion_scores.values()), 1.0, atol=1e-3)
    assert len(res.embedding) == 64


def test_classify_waveform(ser_classifier, synthetic_speech):
    y, sr = synthetic_speech
    res = ser_classifier.classify_waveform(y, sr)

    assert isinstance(res, AcousticSERResult)
    assert res.dominant_emotion in ser_classifier.labels
    assert 0.0 <= res.confidence <= 1.0
    assert np.isclose(sum(res.emotion_scores.values()), 1.0, atol=1e-3)
    assert len(res.embedding) == 64
    assert res.vad is not None
    assert -1.0 <= res.vad.valence <= 1.0
    assert -1.0 <= res.vad.arousal <= 1.0
    assert -1.0 <= res.vad.dominance <= 1.0


def test_classify_chunk_bytes(ser_classifier, synthetic_speech):
    y, sr = synthetic_speech
    # Convert to PCM16 bytes
    pcm_bytes = (y * 32767).astype(np.int16).tobytes()
    res = ser_classifier.classify_chunk(pcm_bytes, sr=sr)

    assert isinstance(res, AcousticSERResult)
    assert res.dominant_emotion in ser_classifier.labels
    assert len(res.embedding) == 64


def test_prosody_heuristic_backend(synthetic_speech):
    y, sr = synthetic_speech
    classifier = DeepSpeechEmotionClassifier(backend="prosody_heuristic")
    res = classifier.classify_waveform(y, sr)

    assert isinstance(res, AcousticSERResult)
    assert res.backend == "prosody_heuristic"
    assert res.dominant_emotion in classifier.labels


def test_extract_embedding(ser_classifier, synthetic_speech):
    y, sr = synthetic_speech
    embedding = ser_classifier.extract_embedding(y, sr)

    assert isinstance(embedding, list)
    assert len(embedding) == 64
    assert all(isinstance(v, float) for v in embedding)


def test_empty_audio_handling(ser_classifier):
    empty_y = np.array([], dtype=np.float32)
    res = ser_classifier.classify_waveform(empty_y, 16000)

    assert res.dominant_emotion == "neutral"
    assert res.duration_sec == 0.0
    assert len(res.embedding) == 64
