"""Unit tests for Acoustic Prosody Extractor and Voice Sentiment Classifier."""

import unittest
import numpy as np
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.core.types import AcousticFeatures, VoiceEmotionResult


class TestAudioPipeline(unittest.TestCase):

    def setUp(self):
        self.extractor = AcousticProsodyExtractor(sample_rate=16000)
        self.classifier = VoiceSentimentClassifier()

    def test_silence_audio_handling(self):
        silent_audio = np.zeros(16000, dtype=np.float32)
        features = self.extractor.extract_features(silent_audio)
        self.assertFalse(features.speech_active)
        self.assertAlmostEqual(features.rms_energy, 0.0)

    def test_sine_wave_pitch_detection(self):
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        target_f0 = 220.0  # 220 Hz pitch
        audio = 0.5 * np.sin(2 * np.pi * target_f0 * t).astype(np.float32)

        features = self.extractor.extract_features(audio)
        self.assertTrue(features.speech_active)
        self.assertGreater(features.pitch_hz, 0.0)
        # Should be close to 220 Hz
        self.assertAlmostEqual(features.pitch_hz, target_f0, delta=15.0)

    def test_voice_sentiment_classification(self):
        acoustics = AcousticFeatures(
            pitch_hz=250.0,
            pitch_confidence=0.8,
            rms_energy=0.08,
            jitter_percent=0.01,
            shimmer_percent=0.03,
            speech_active=True
        )
        res = self.classifier.classify_voice_emotion(acoustics)
        self.assertIsInstance(res, VoiceEmotionResult)
        self.assertIn(res.dominant_emotion, ["joy", "surprise", "neutral", "anger", "fear", "sadness", "disgust", "contempt"])
        self.assertGreaterEqual(res.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
