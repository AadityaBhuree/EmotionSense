"""Tests for AcousticDiarizer speaker diarization engine."""

import io
import numpy as np
import soundfile as sf
from src.audio.diarizer import AcousticDiarizer
from src.core.types import DiarizationResult


def _generate_tone_burst(freq: float, duration: float, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    """Generates a synthetic harmonic tone burst with slight frequency variation."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental + harmonics for rich acoustic features
    signal = (
        amp * 0.6 * np.sin(2 * np.pi * freq * t)
        + amp * 0.3 * np.sin(2 * np.pi * (freq * 2) * t)
        + amp * 0.1 * np.sin(2 * np.pi * (freq * 3) * t)
    )
    # Apply fade in / fade out to avoid clicks
    fade_len = int(sr * 0.02)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    signal[:fade_len] *= fade_in
    signal[-fade_len:] *= fade_out
    return signal.astype(np.float32)


def test_diarizer_initialization():
    diarizer = AcousticDiarizer(sample_rate=16000, num_speakers=2)
    assert diarizer.sample_rate == 16000
    assert diarizer.num_speakers == 2
    assert diarizer.min_speech_duration == 0.4


def test_empty_and_silence_audio():
    diarizer = AcousticDiarizer()
    res = diarizer.diarize(None)
    assert isinstance(res, DiarizationResult)
    assert len(res.turns) == 0

    silence = np.zeros(16000 * 2, dtype=np.float32)
    res_silence = diarizer.diarize(silence)
    assert len(res_silence.turns) == 0
    assert res_silence.total_audio_duration == 2.0


def test_synthetic_single_speaker():
    diarizer = AcousticDiarizer(sample_rate=16000, num_speakers=1)
    # 0.5s silence + 1.2s tone (150Hz) + 0.5s silence
    tone = _generate_tone_burst(freq=150.0, duration=1.2, sr=16000)
    audio = np.concatenate([np.zeros(8000, dtype=np.float32), tone, np.zeros(8000, dtype=np.float32)])

    res = diarizer.diarize(audio)
    assert len(res.turns) >= 1
    assert len(res.speakers) == 1
    assert res.speakers[0] == "Speaker_0"
    assert res.total_speech_duration > 0.8
    assert res.dominance_ratios["Speaker_0"] == 1.0


def test_synthetic_two_speakers_alternating():
    diarizer = AcousticDiarizer(sample_rate=16000, num_speakers=2)
    sr = 16000

    # Speaker A: 120 Hz, 1.2s duration
    spk_a = _generate_tone_burst(freq=120.0, duration=1.2, sr=sr)
    # Silence: 0.6s
    silence = np.zeros(int(sr * 0.6), dtype=np.float32)
    # Speaker B: 280 Hz, 1.5s duration
    spk_b = _generate_tone_burst(freq=280.0, duration=1.5, sr=sr)

    conversation = np.concatenate([silence, spk_a, silence, spk_b, silence])

    res = diarizer.diarize(conversation)
    assert len(res.turns) >= 2
    assert len(res.speakers) == 2
    assert "Speaker_0" in res.speakers
    assert "Speaker_1" in res.speakers
    # Dominance ratios should sum approximately to 1.0
    total_ratio = sum(res.dominance_ratios.values())
    assert abs(total_ratio - 1.0) < 0.05
    # Verify turns contain acoustics
    for turn in res.turns:
        assert turn.duration > 0
        assert turn.acoustics is not None


def test_diarize_file_bytes():
    diarizer = AcousticDiarizer(sample_rate=16000, num_speakers=2)
    sr = 16000

    spk_a = _generate_tone_burst(freq=130.0, duration=1.0, sr=sr)
    spk_b = _generate_tone_burst(freq=260.0, duration=1.0, sr=sr)
    audio = np.concatenate([spk_a, np.zeros(int(sr * 0.5), dtype=np.float32), spk_b])

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    wav_bytes = buf.getvalue()

    res = diarizer.diarize_file(wav_bytes)
    assert len(res.turns) >= 2
    assert len(res.speakers) == 2
    assert res.total_speech_duration > 1.5
