"""Unit tests for LiveSpeechTranscriber and phonetic affect alignment."""

import numpy as np
from src.audio.speech_transcriber import LiveSpeechTranscriber, LiveSpeechTranscriptionResult


def test_speech_transcriber_initialization():
    transcriber = LiveSpeechTranscriber()
    assert isinstance(transcriber.is_available, bool)


def test_transcribe_text_stream_joy():
    transcriber = LiveSpeechTranscriber()
    res = transcriber.transcribe_text_stream("I am overjoyed and extremely happy with this milestone! 🎉")
    assert isinstance(res, LiveSpeechTranscriptionResult)
    assert res.full_transcript != ""
    assert res.text_emotion is not None
    assert res.text_emotion.dominant_emotion == "joy"
    assert len(res.tokens) > 0


def test_transcribe_text_stream_empty():
    transcriber = LiveSpeechTranscriber()
    res = transcriber.transcribe_text_stream("")
    assert res.full_transcript == ""
    assert len(res.tokens) == 0


def test_process_audio_buffer_silence():
    transcriber = LiveSpeechTranscriber()
    silence = np.zeros(1600, dtype=np.float32)
    res = transcriber.process_audio_buffer(silence)
    assert res.full_transcript == ""


def test_process_audio_buffer_voice():
    transcriber = LiveSpeechTranscriber()
    t = np.linspace(0, 0.2, 3200)
    sine_wave = (0.25 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    res = transcriber.process_audio_buffer(sine_wave)
    assert res.timestamp > 0


def test_transcribe_audio_bytes_empty():
    transcriber = LiveSpeechTranscriber()
    res = transcriber.transcribe_audio_bytes(b"")
    assert res.full_transcript == ""
    assert res.tokens == []


def test_transcribe_audio_bytes_mocked_success(monkeypatch):
    from unittest.mock import MagicMock
    import io
    import soundfile as sf
    from src.core.types import VoiceEmotionResult, AcousticFeatures

    transcriber = LiveSpeechTranscriber()
    transcriber._is_available = True
    mock_recognizer = MagicMock()
    mock_recognizer.record.return_value = MagicMock()
    mock_recognizer.recognize_google.return_value = "hello world I am very happy"
    transcriber._recognizer = mock_recognizer

    # Create dummy WAV bytes
    buf = io.BytesIO()
    dummy_audio = np.zeros(16000, dtype=np.float32)
    sf.write(buf, dummy_audio, 16000, format="WAV")
    wav_bytes = buf.getvalue()

    voice_res = VoiceEmotionResult(
        dominant_emotion="joy",
        confidence=0.9,
        acoustics=AcousticFeatures(pitch_hz=210.5, rms_energy=0.08)
    )

    res = transcriber.transcribe_audio_bytes(wav_bytes, voice_result=voice_res)
    assert res.full_transcript == "hello world I am very happy"
    assert res.text_emotion is not None
    assert res.voice_emotion is voice_res
    assert len(res.tokens) == 6
    assert res.tokens[0].word == "hello"
    assert res.tokens[0].pitch_hz == 210.5
    assert res.tokens[0].rms_energy == 0.08


def test_transcribe_audio_array_mocked_success():
    from unittest.mock import MagicMock
    from src.core.types import VoiceEmotionResult, AcousticFeatures

    transcriber = LiveSpeechTranscriber()
    transcriber._is_available = True
    mock_recognizer = MagicMock()
    mock_recognizer.recognize_google.return_value = "this is an amazing breakthrough"
    transcriber._recognizer = mock_recognizer

    t = np.linspace(0, 0.5, 8000)
    audio = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    voice_res = VoiceEmotionResult(
        dominant_emotion="joy",
        confidence=0.92,
        acoustics=AcousticFeatures(pitch_hz=300.0, rms_energy=0.15)
    )

    res = transcriber.transcribe_audio_array(audio, sample_rate=16000, voice_result=voice_res)
    assert res.full_transcript == "this is an amazing breakthrough"
    assert res.text_emotion is not None
    assert len(res.tokens) == 5
    assert res.tokens[0].pitch_hz == 300.0


def test_transcribe_audio_recognition_exception_fallback():
    from unittest.mock import MagicMock

    transcriber = LiveSpeechTranscriber()
    transcriber._is_available = True
    mock_recognizer = MagicMock()
    mock_recognizer.recognize_google.side_effect = RuntimeError("Service unreachable")
    transcriber._recognizer = mock_recognizer

    t = np.linspace(0, 0.2, 3200)
    audio = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    res = transcriber.transcribe_audio_array(audio)
    assert res.full_transcript == ""
    assert res.tokens == []

