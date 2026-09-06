"""Acoustic tone and vocal sentiment classification engine."""

import numpy as np
from typing import Dict, Any
from src.core.types import VoiceEmotionResult, AcousticFeatures, AffectVector
from src.core.config import EMOTION_LABELS, AUDIO_THRESHOLDS, EMOTION_VAD_COORDINATES


class VoiceSentimentClassifier:
    """Classifies affective voice tone into categorical emotion probabilities and vocal stress levels."""

    def __init__(self):
        self.labels = EMOTION_LABELS

    def _compute_affect_vector(self, probabilities: Dict[str, float]) -> AffectVector:
        valence = 0.0
        arousal = 0.0
        dominance = 0.0
        for emo, p in probabilities.items():
            vad = EMOTION_VAD_COORDINATES.get(emo, [0.0, 0.0, 0.0])
            valence += p * vad[0]
            arousal += p * vad[1]
            dominance += p * vad[2]
        return AffectVector(
            valence=float(np.clip(valence, -1.0, 1.0)),
            arousal=float(np.clip(arousal, -1.0, 1.0)),
            dominance=float(np.clip(dominance, -1.0, 1.0)),
        )

    def classify_voice_emotion(self, acoustics: AcousticFeatures) -> VoiceEmotionResult:
        """Translates acoustic metrics into emotion probabilities and vocal stress level."""
        if not acoustics.speech_active:
            neutral_probs = {"neutral": 1.0, "joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "contempt": 0.0}
            return VoiceEmotionResult(
                dominant_emotion="neutral",
                confidence=0.85,
                probabilities=neutral_probs,
                acoustics=acoustics,
                affect=self._compute_affect_vector(neutral_probs),
                vocal_stress_level=0.0,
            )

        pitch = acoustics.pitch_hz
        rms = acoustics.rms_energy
        jitter = acoustics.jitter_percent
        shimmer = acoustics.shimmer_percent
        zcr = acoustics.zero_crossing_rate

        # Acoustic heuristics & affective prosody rules
        # High Pitch + High Energy -> Joy or Surprise
        # High Pitch + High Jitter/Shimmer + High Energy -> Anger or Fear
        # Low Pitch + Low Energy + Flat Tempo -> Sadness
        
        raw_scores = {
            "joy": (pitch / 250.0) * 0.8 + (rms * 15.0) * 0.7 + (1.0 - jitter * 10.0) * 0.3,
            "sadness": (1.0 - min(pitch / 200.0, 1.0)) * 1.2 + (1.0 - min(rms * 20.0, 1.0)) * 1.0,
            "anger": (rms * 25.0) * 1.1 + (jitter * 20.0) * 0.9 + (zcr * 5.0) * 0.6,
            "fear": (pitch / 280.0) * 0.9 + (jitter * 25.0) * 1.2 + (shimmer * 15.0) * 0.8,
            "surprise": (pitch / 260.0) * 1.3 + (rms * 18.0) * 0.8,
            "disgust": (1.0 - min(pitch / 180.0, 1.0)) * 0.7 + (zcr * 4.0) * 0.5,
            "neutral": 0.55,
            "contempt": (1.0 - min(rms * 15.0, 1.0)) * 0.5 + (jitter * 10.0) * 0.4,
        }

        # Softmax normalization
        exp_scores = {k: np.exp(v * 2.2) for k, v in raw_scores.items()}
        total = sum(exp_scores.values())
        probabilities = {k: float(v / total) for k, v in exp_scores.items()}

        dominant_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[dominant_emotion]

        # Calculate composite vocal stress index
        vocal_stress = float(np.clip(
            (jitter / AUDIO_THRESHOLDS["jitter_stress_threshold"]) * 0.4 +
            (shimmer / AUDIO_THRESHOLDS["shimmer_stress_threshold"]) * 0.4 +
            (rms * 5.0) * 0.2,
            0.0, 1.0
        ))

        affect = self._compute_affect_vector(probabilities)

        return VoiceEmotionResult(
            dominant_emotion=dominant_emotion,
            confidence=confidence,
            probabilities=probabilities,
            acoustics=acoustics,
            affect=affect,
            vocal_stress_level=vocal_stress,
        )

    def classify_prosody(self, acoustics: AcousticFeatures) -> VoiceEmotionResult:
        """Alias for classify_voice_emotion."""
        return self.classify_voice_emotion(acoustics)

    def classify(self, acoustics: AcousticFeatures) -> VoiceEmotionResult:
        """Alias for classify_voice_emotion."""
        return self.classify_voice_emotion(acoustics)
