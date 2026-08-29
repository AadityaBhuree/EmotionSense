"""WebRTC Video and Audio frame processor for live stream inference."""

import av
import cv2
import numpy as np
import threading
from typing import Optional

from src.vision.face_mesh import FaceMeshDetector
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.core.types import MultimodalEmotionState, VisionEmotionResult, VoiceEmotionResult

try:
    from streamlit_webrtc import VideoTransformerBase, AudioProcessorBase
except ImportError:
    class VideoTransformerBase:
        pass
    class AudioProcessorBase:
        pass


class MultimodalVideoProcessor(VideoTransformerBase):
    """Processes real-time video frames, overlays 468-point 3D face mesh, and updates shared state."""

    def __init__(self):
        self.face_mesh = FaceMeshDetector()
        self.emotion_classifier = FacialEmotionClassifier()
        self.audio_extractor = AcousticProsodyExtractor()
        self.voice_classifier = VoiceSentimentClassifier()
        self.fusion_engine = MultimodalFusionEngine(window_size=20)
        
        self.lock = threading.Lock()
        self.latest_state: Optional[MultimodalEmotionState] = None
        self.latest_vision: Optional[VisionEmotionResult] = None
        self.latest_voice: Optional[VoiceEmotionResult] = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Processes incoming video frame and annotates facial mesh landmarks."""
        img = frame.to_ndarray(format="bgr24")

        # 1. Process facial mesh
        landmarks, head_pose = self.face_mesh.process_frame(img)
        vision_res = self.emotion_classifier.classify_emotion(landmarks, head_pose)

        # 2. Draw mesh overlay if face detected
        if landmarks is not None:
            img = self.face_mesh.draw_mesh_overlay(img, landmarks)

            # Draw dominant emotion tag on canvas
            emo_text = f"{vision_res.dominant_emotion.upper()} ({int(vision_res.confidence * 100)}%)"
            cv2.putText(
                img, emo_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (99, 102, 241), 2, cv2.LINE_AA
            )

        # 3. Fuse modalities and store latest state
        fused_state = self.fusion_engine.fuse(vision_res, self.latest_voice)
        with self.lock:
            self.latest_vision = vision_res
            self.latest_state = fused_state

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_latest_state(self) -> Optional[MultimodalEmotionState]:
        with self.lock:
            return self.latest_state
