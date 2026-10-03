"""Acoustic prosody, voice emotion, and live speech transcription module."""

from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.audio.speech_transcriber import LiveSpeechTranscriber
from src.audio.diarizer import AcousticDiarizer

try:
    from src.audio.deep_ser import DeepSpeechEmotionClassifier
except (ImportError, Exception):
    DeepSpeechEmotionClassifier = None  # type: ignore

__all__ = [
    "AcousticProsodyExtractor",
    "VoiceSentimentClassifier",
    "LiveSpeechTranscriber",
    "AcousticDiarizer",
    "DeepSpeechEmotionClassifier",
]

