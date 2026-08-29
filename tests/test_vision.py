"""Unit tests for Vision Face Mesh and Facial Emotion Classifier."""

import unittest
import numpy as np
from src.vision.face_mesh import FaceMeshDetector
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.core.types import VisionEmotionResult, FacialActionUnits


class TestVisionPipeline(unittest.TestCase):

    def setUp(self):
        self.detector = FaceMeshDetector()
        self.classifier = FacialEmotionClassifier()

    def test_null_frame_handling(self):
        landmarks, pose = self.detector.process_frame(None)
        self.assertIsNone(landmarks)
        self.assertEqual(pose["yaw"], 0.0)

    def test_empty_landmarks_classification(self):
        res = self.classifier.classify_emotion(None, {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
        self.assertFalse(res.face_detected)
        self.assertIsInstance(res, VisionEmotionResult)

    def test_action_units_bounds(self):
        # Create synthetic 468 landmarks normalized
        synthetic_landmarks = np.zeros((468, 3), dtype=np.float32)
        # Set outer eye corners for scale
        synthetic_landmarks[33] = [0.3, 0.4, 0.0]
        synthetic_landmarks[263] = [0.7, 0.4, 0.0]

        aus = self.classifier.extract_action_units(synthetic_landmarks)
        self.assertIsInstance(aus, FacialActionUnits)
        self.assertGreaterEqual(aus.lip_corner_puller, 0.0)
        self.assertLessEqual(aus.lip_corner_puller, 1.0)


if __name__ == "__main__":
    unittest.main()
