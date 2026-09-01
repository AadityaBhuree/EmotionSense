try:
    import av
except ImportError:
    av = None

import cv2
import numpy as np
import threading
from typing import Optional, Any

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
    """Processes real-time video frames, overlays 468-point 3D face mesh, and renders live affective HUD."""

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
        self.live_text_prompt: Optional[str] = None

    def set_live_text(self, text: Optional[str]):
        """Sets active text/spoken prompt to fuse with real-time video stream."""
        with self.lock:
            self.live_text_prompt = text

    def recv(self, frame: Any) -> Any:
        """Processes incoming video frame and annotates facial mesh landmarks and affective HUD."""
        if av is None or not hasattr(frame, "to_ndarray"):
            return frame

        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # 1. Process facial mesh
        landmarks, head_pose = self.face_mesh.process_frame(img)
        vision_res = self.emotion_classifier.classify_emotion(landmarks, head_pose)

        # 2. Draw mesh overlay if face detected
        if landmarks is not None:
            img = self.face_mesh.draw_mesh_overlay(img, landmarks)

            # Draw Dominant Emotion HUD Card on top-left
            emo = vision_res.dominant_emotion.upper()
            conf_pct = int(vision_res.confidence * 100)
            
            # Background header pill
            cv2.rectangle(img, (15, 15), (280, 75), (15, 23, 42), -1)
            cv2.rectangle(img, (15, 15), (280, 75), (99, 102, 241), 1)

            cv2.putText(
                img, f"AFFECT: {emo}", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (248, 250, 252), 2, cv2.LINE_AA
            )
            cv2.putText(
                img, f"CONF: {conf_pct}%  POSE: Y:{int(head_pose['yaw'])} P:{int(head_pose['pitch'])}", (25, 63),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA
            )

        # 3. Render Live Subtitle Bar if live text is active
        with self.lock:
            prompt = self.live_text_prompt

        if prompt:
            # Subtitle background bar at bottom
            cv2.rectangle(img, (20, h - 50), (w - 20, h - 15), (15, 23, 42), -1)
            cv2.rectangle(img, (20, h - 50), (w - 20, h - 15), (6, 182, 212), 1)
            sub_text = prompt[:50] + ("..." if len(prompt) > 50 else "")
            cv2.putText(
                img, f"SPEECH: {sub_text}", (30, h - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 250, 252), 1, cv2.LINE_AA
            )

        # 4. Fuse modalities and store latest state
        fused_state = self.fusion_engine.fuse(vision_res, self.latest_voice)
        with self.lock:
            self.latest_vision = vision_res
            self.latest_state = fused_state

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_latest_state(self) -> Optional[MultimodalEmotionState]:
        with self.lock:
            return self.latest_state

