"""WebRTC Real-Time Audio & Video Stream Processing Engine.

Synchronizes MediaPipe 3D FaceMesh video transformations with real-time PyAV
microphone audio prosody extraction and multimodal affect late fusion.
"""

try:
    import av
except ImportError:
    av = None

import cv2
import numpy as np
import threading
import collections
from typing import Optional, Any, Dict, List

from src.vision.face_mesh import FaceMeshDetector
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.core.types import (
    MultimodalEmotionState,
    VisionEmotionResult,
    VoiceEmotionResult,
    AcousticFeatures,
)

try:
    from streamlit_webrtc import VideoTransformerBase, AudioProcessorBase
except ImportError:
    class VideoTransformerBase:
        pass
    class AudioProcessorBase:
        pass


class MultimodalStreamContext:
    """Thread-safe state synchronization bus between concurrent WebRTC video and audio worker threads."""

    def __init__(self, window_size: int = 30):
        self.lock = threading.Lock()
        self.fusion_engine = MultimodalFusionEngine(window_size=window_size)
        self.latest_state: Optional[MultimodalEmotionState] = None
        self.latest_vision: Optional[VisionEmotionResult] = None
        self.latest_voice: Optional[VoiceEmotionResult] = None
        self.latest_acoustics: Optional[AcousticFeatures] = None
        self.live_text_prompt: Optional[str] = None
        self.speech_active: bool = False
        self.audio_energy_history: collections.deque = collections.deque(maxlen=30)
        self.sample_count: int = 0

    def update_voice(self, voice_res: VoiceEmotionResult, acoustics: AcousticFeatures):
        """Called by AudioProcessor when a new audio chunk is classified."""
        with self.lock:
            self.latest_voice = voice_res
            self.latest_acoustics = acoustics
            self.speech_active = acoustics.speech_active
            self.audio_energy_history.append(acoustics.rms_energy)

    def update_state(self, state: MultimodalEmotionState, vision_res: VisionEmotionResult):
        """Called by VideoProcessor after late fusion."""
        with self.lock:
            self.latest_state = state
            self.latest_vision = vision_res
            self.sample_count += 1

    def set_live_text(self, text: Optional[str]):
        """Injects active text/spoken prompt into the fusion pipeline."""
        with self.lock:
            self.live_text_prompt = text

    def get_latest_state(self) -> Optional[MultimodalEmotionState]:
        with self.lock:
            return self.latest_state

    def get_latest_voice(self) -> Optional[VoiceEmotionResult]:
        with self.lock:
            return self.latest_voice

    def get_latest_acoustics(self) -> Optional[AcousticFeatures]:
        with self.lock:
            return self.latest_acoustics

    def get_live_text(self) -> Optional[str]:
        with self.lock:
            return self.live_text_prompt


class MultimodalAudioProcessor(AudioProcessorBase):
    """Processes real-time WebRTC microphone audio buffers.

    Resamples incoming audio frames into 16kHz mono float32, maintains a rolling buffer,
    and calculates continuous acoustic prosody and vocal sentiment.
    """

    def __init__(self, context: Optional[MultimodalStreamContext] = None, sample_rate: int = 16000):
        self.context = context or MultimodalStreamContext()
        self.sample_rate = sample_rate
        self.audio_extractor = AcousticProsodyExtractor(sample_rate=self.sample_rate)
        self.voice_classifier = VoiceSentimentClassifier()

        # Rolling circular buffer for 1.5 seconds of audio (24,000 samples at 16kHz)
        self.buffer_size = int(self.sample_rate * 1.5)
        self.audio_buffer = collections.deque(maxlen=self.buffer_size)
        self.resampler = None
        self._step_counter = 0

    def recv(self, frame: Any) -> Any:
        """Processes incoming audio frame from WebRTC microphone stream."""
        if av is None or not hasattr(frame, "to_ndarray"):
            return frame

        try:
            # Initialize lazy resampler to target format (fltp, mono, 16000Hz)
            if self.resampler is None:
                self.resampler = av.audio.resampler.AudioResampler(
                    format="fltp", layout="mono", rate=self.sample_rate
                )

            resampled_frames = self.resampler.resample(frame)
            for rf in resampled_frames:
                arr = rf.to_ndarray()
                if arr is not None and len(arr) > 0:
                    samples_1d = arr[0].astype(np.float32)
                    self.audio_buffer.extend(samples_1d)

            self._step_counter += 1
            # Run acoustic feature extraction every ~4 audio frames (approx 80-120ms)
            if self._step_counter % 4 == 0 and len(self.audio_buffer) >= int(self.sample_rate * 0.25):
                buf_array = np.array(self.audio_buffer, dtype=np.float32)
                acoustics = self.audio_extractor.extract_features(buf_array)
                voice_res = self.voice_classifier.classify_voice_emotion(acoustics)
                self.context.update_voice(voice_res, acoustics)

        except Exception:
            pass

        return frame


class MultimodalVideoProcessor(VideoTransformerBase):
    """Processes real-time video frames, overlays 468-point 3D face mesh, and renders live affective HUD."""

    def __init__(self, context: Optional[MultimodalStreamContext] = None):
        self.context = context or MultimodalStreamContext()
        self.face_mesh = FaceMeshDetector()
        self.emotion_classifier = FacialEmotionClassifier()
        self.fusion_engine = self.context.fusion_engine

    def set_live_text(self, text: Optional[str]):
        """Sets active text/spoken prompt to fuse with real-time video stream."""
        self.context.set_live_text(text)

    def recv(self, frame: Any) -> Any:
        """Processes incoming video frame and annotates facial mesh landmarks and affective HUD."""
        if av is None or not hasattr(frame, "to_ndarray"):
            return frame

        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # 1. Process facial mesh
        landmarks, head_pose = self.face_mesh.process_frame(img)
        vision_res = self.emotion_classifier.classify_emotion(landmarks, head_pose)

        # 2. Retrieve concurrent acoustic telemetry from shared stream context
        voice_res = self.context.get_latest_voice()
        acoustics = self.context.get_latest_acoustics()

        # 3. Draw mesh overlay if face detected
        if landmarks is not None:
            img = self.face_mesh.draw_mesh_overlay(img, landmarks)

            # Draw Dominant Affect HUD Card on top-left
            emo = vision_res.dominant_emotion.upper()
            conf_pct = int(vision_res.confidence * 100)

            # Left card: Vision Affect
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

        # 4. Render Acoustic Telemetry HUD on top-right if audio is active
        if acoustics is not None:
            card_w = 260
            rx1 = max(w - card_w - 15, 300)
            rx2 = w - 15
            cv2.rectangle(img, (rx1, 15), (rx2, 75), (15, 23, 42), -1)

            border_color = (16, 185, 129) if acoustics.speech_active else (100, 116, 139)
            cv2.rectangle(img, (rx1, 15), (rx2, 75), border_color, 1)

            mic_status = "MIC: LIVE" if acoustics.speech_active else "MIC: IDLE"
            v_emo = voice_res.dominant_emotion.upper() if voice_res and acoustics.speech_active else "QUIET"
            stress_pct = int(voice_res.vocal_stress_level * 100) if voice_res else 0

            cv2.putText(
                img, f"{mic_status} | {v_emo}", (rx1 + 10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (248, 250, 252), 2, cv2.LINE_AA
            )
            cv2.putText(
                img, f"F0:{int(acoustics.pitch_hz)}Hz  STRESS:{stress_pct}%", (rx1 + 10, 63),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA
            )

        # 5. Render Live Subtitle Bar if live text prompt is active
        prompt = self.context.get_live_text()
        if prompt:
            cv2.rectangle(img, (20, h - 50), (w - 20, h - 15), (15, 23, 42), -1)
            cv2.rectangle(img, (20, h - 50), (w - 20, h - 15), (6, 182, 212), 1)
            sub_text = prompt[:50] + ("..." if len(prompt) > 50 else "")
            cv2.putText(
                img, f"SPEECH: {sub_text}", (30, h - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 250, 252), 1, cv2.LINE_AA
            )

        # 6. Fuse modalities and store latest synchronized state
        fused_state = self.fusion_engine.fuse(vision=vision_res, voice=voice_res)
        self.context.update_state(fused_state, vision_res)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_latest_state(self) -> Optional[MultimodalEmotionState]:
        """Provides backwards compatibility for callers accessing get_latest_state directly."""
        return self.context.get_latest_state()
