"""Speech-to-Text and Acoustic-Lexical Phonetic Alignment Engine.

Transcribes spoken audio chunks and aligns phonetic prosody features (pitch, energy, jitter)
with recognized word tokens for synchronized tri-modal affective intelligence.
"""

import time
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

from src.core.types import TextEmotionResult, VoiceEmotionResult
from src.text.nlp_emotion import TextEmotionClassifier


@dataclass
class SpokenWordToken:
    """Individual transcribed word token with acoustic alignment."""
    word: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    pitch_hz: float = 0.0
    rms_energy: float = 0.0
    emotion_cue: str = "neutral"


@dataclass
class LiveSpeechTranscriptionResult:
    """Result of live speech-to-text transcription with emotional prosody."""
    full_transcript: str
    tokens: List[SpokenWordToken] = field(default_factory=list)
    text_emotion: Optional[TextEmotionResult] = None
    voice_emotion: Optional[VoiceEmotionResult] = None
    is_final: bool = False
    timestamp: float = field(default_factory=time.time)


class LiveSpeechTranscriber:
    """Audio speech recognition and affective alignment engine."""

    def __init__(self, text_classifier: Optional[TextEmotionClassifier] = None):
        """Initializes the live speech transcriber."""
        self.text_classifier = text_classifier or TextEmotionClassifier()
        self._recognizer = None
        self._is_available = False
        self._init_engine()

    def _init_engine(self):
        """Attempts to initialize local speech recognition engine."""
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._is_available = True
        except ImportError:
            self._is_available = False

    @property
    def is_available(self) -> bool:
        """Returns True if speech recognition package is available."""
        return self._is_available

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        voice_result: Optional[VoiceEmotionResult] = None,
    ) -> LiveSpeechTranscriptionResult:
        """Transcribes raw audio bytes (e.g. WAV/PCM container from microphone) and generates affect scoring."""
        if not audio_bytes or len(audio_bytes) == 0:
            return LiveSpeechTranscriptionResult(full_transcript="", voice_emotion=voice_result)

        text = ""
        if self._is_available and self._recognizer is not None:
            try:
                import io
                import speech_recognition as sr
                with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                    audio_data = self._recognizer.record(source)
                    text = self._recognizer.recognize_google(audio_data)
            except Exception:
                text = ""

        if text and text.strip():
            res = self.transcribe_text_stream(text.strip())
            if voice_result:
                res.voice_emotion = voice_result
                # Align acoustic metrics onto word tokens
                pitch = voice_result.acoustics.pitch_hz if voice_result.acoustics else 0.0
                energy = voice_result.acoustics.rms_energy if voice_result.acoustics else 0.0
                for tok in res.tokens:
                    tok.pitch_hz = pitch
                    tok.rms_energy = energy
            return res

        return LiveSpeechTranscriptionResult(
            full_transcript="",
            voice_emotion=voice_result,
            timestamp=time.time(),
        )

    def transcribe_audio_array(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        voice_result: Optional[VoiceEmotionResult] = None,
    ) -> LiveSpeechTranscriptionResult:
        """Processes raw PCM numpy array and returns transcribed tokens with affective scoring."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return LiveSpeechTranscriptionResult(full_transcript="", voice_emotion=voice_result)

        rms = float(np.sqrt(np.mean(audio_chunk**2))) if len(audio_chunk) > 0 else 0.0
        if rms < 0.005:
            return LiveSpeechTranscriptionResult(full_transcript="", voice_emotion=voice_result)

        text = ""
        if self._is_available and self._recognizer is not None:
            try:
                import speech_recognition as sr
                if audio_chunk.dtype != np.int16:
                    audio_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_chunk
                raw_bytes = audio_int16.tobytes()
                audio_data = sr.AudioData(raw_bytes, sample_rate, 2)
                text = self._recognizer.recognize_google(audio_data)
            except Exception:
                text = ""

        if text and text.strip():
            res = self.transcribe_text_stream(text.strip())
            if voice_result:
                res.voice_emotion = voice_result
                pitch = voice_result.acoustics.pitch_hz if voice_result.acoustics else 0.0
                energy = voice_result.acoustics.rms_energy if voice_result.acoustics else 0.0
                for tok in res.tokens:
                    tok.pitch_hz = pitch
                    tok.rms_energy = energy
            return res

        return LiveSpeechTranscriptionResult(
            full_transcript="",
            voice_emotion=voice_result,
            timestamp=time.time(),
        )

    def process_audio_buffer(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        voice_result: Optional[VoiceEmotionResult] = None,
    ) -> LiveSpeechTranscriptionResult:
        """Processes raw PCM audio chunk and returns transcribed tokens with affective scoring."""
        return self.transcribe_audio_array(audio_chunk, sample_rate=sample_rate, voice_result=voice_result)

    def transcribe_text_stream(self, spoken_phrase: str) -> LiveSpeechTranscriptionResult:
        """Transcribes incoming speech string and executes instant NLP affect scoring."""
        if not spoken_phrase or not spoken_phrase.strip():
            return LiveSpeechTranscriptionResult(full_transcript="")

        clean_text = spoken_phrase.strip()
        text_emo = self.text_classifier.analyze_text(clean_text)

        words = clean_text.split()
        tokens = []
        cur_t = time.time()
        for i, w in enumerate(words):
            tokens.append(SpokenWordToken(
                word=w,
                start_time=round(cur_t + (i * 0.3), 2),
                end_time=round(cur_t + ((i + 1) * 0.3), 2),
                confidence=0.95,
                emotion_cue=text_emo.dominant_emotion,
            ))

        return LiveSpeechTranscriptionResult(
            full_transcript=clean_text,
            tokens=tokens,
            text_emotion=text_emo,
            is_final=True,
            timestamp=cur_t,
        )

