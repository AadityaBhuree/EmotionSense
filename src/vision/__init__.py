"""Computer vision and facial affective intelligence module."""

from src.vision.face_mesh import FaceMeshDetector
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.vision.multi_face_tracker import MultiFaceTracker

__all__ = ["FaceMeshDetector", "FacialEmotionClassifier", "MultiFaceTracker"]
