"""Unit tests for Multimodal Late Fusion Engine and Affect Metrics."""

import unittest
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.fusion.metrics import AffectMetricsCalculator
from src.core.types import (
    VisionEmotionResult,
    VoiceEmotionResult,
    FacialActionUnits,
    AcousticFeatures,
    MultimodalEmotionState,
)


class TestMultimodalFusion(unittest.TestCase):

    def setUp(self):
        self.engine = MultimodalFusionEngine(window_size=10)
        self.calc = AffectMetricsCalculator()

    def test_fusion_with_vision_only(self):
        vision = VisionEmotionResult(
            face_detected=True,
            dominant_emotion="joy",
            confidence=0.9,
            probabilities={"joy": 0.9, "neutral": 0.1, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "contempt": 0.0},
            action_units=FacialActionUnits(lip_corner_puller=0.8),
            head_pose={"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        )

        state = self.engine.fuse(vision, None)
        self.assertIsInstance(state, MultimodalEmotionState)
        self.assertEqual(state.dominant_emotion, "joy")
        self.assertGreater(state.affect.valence, 0.0)  # Joy has positive valence
        self.assertGreater(state.attention_score, 0.9)  # Centered head pose

    def test_attention_score_penalty(self):
        # Direct gaze vs turned head
        centered_vision = VisionEmotionResult(
            face_detected=True,
            head_pose={"yaw": 2.0, "pitch": 1.0, "roll": 0.0}
        )
        turned_vision = VisionEmotionResult(
            face_detected=True,
            head_pose={"yaw": 30.0, "pitch": 15.0, "roll": 10.0}
        )

        att_centered = self.calc.calculate_attention_score(centered_vision)
        att_turned = self.calc.calculate_attention_score(turned_vision)
        self.assertGreater(att_centered, att_turned)

    def test_vad_quadrant_output(self):
        from src.core.types import AffectVector
        pos_affect = AffectVector(valence=0.8, arousal=0.5, dominance=0.3)
        self.assertIn("High Positive", pos_affect.quadrant())

        neg_affect = AffectVector(valence=-0.7, arousal=0.6, dominance=-0.2)
        self.assertIn("High Negative", neg_affect.quadrant())


if __name__ == "__main__":
    unittest.main()
