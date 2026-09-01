"""Unit tests for LiveSpeechTranscriber and phonetic affect alignment."""

import numpy as np
import pytest
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
