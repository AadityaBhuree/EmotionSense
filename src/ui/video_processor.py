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
import time
from typing import Optional, Any, Dict, List

from src.vision.face_mesh import FaceMeshDetector
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.vision.multi_face_tracker import MultiFaceTracker
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.audio.speech_transcriber import LiveSpeechTranscriber
from src.text.nlp_emotion import TextEmotionClassifier
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.core.types import (
    MultimodalEmotionState,
    VisionEmotionResult,
    VoiceEmotionResult,
    AcousticFeatures,
    TextEmotionResult,
    MultiFaceResult,
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
        self.text_classifier = TextEmotionClassifier()
        self.latest_state: Optional[MultimodalEmotionState] = None
        self.latest_vision: Optional[VisionEmotionResult] = None
        self.latest_voice: Optional[VoiceEmotionResult] = None
        self.latest_acoustics: Optional[AcousticFeatures] = None
        self.live_text_prompt: Optional[str] = None
        self.latest_transcript: Optional[str] = None
        self.latest_text_emotion: Optional[TextEmotionResult] = None
        self.transcript_history: collections.deque = collections.deque(maxlen=25)
        self.speech_active: bool = False
        self.audio_energy_history: collections.deque = collections.deque(maxlen=30)
        self.latest_multi_face: Optional[MultiFaceResult] = None
        self.sample_count: int = 0

    def update_multi_face(self, multi_res: MultiFaceResult):
        """Called by VideoProcessor when multi-face tracking updates."""
        with self.lock:
            self.latest_multi_face = multi_res

    def get_latest_multi_face(self) -> Optional[MultiFaceResult]:
        with self.lock:
            return self.latest_multi_face

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

    def update_transcript(self, text: str, text_res: Optional[TextEmotionResult] = None):
        """Updates live transcript detected by AudioProcessor speech recognizer."""
        with self.lock:
            self.latest_transcript = text
            if text:
                if text_res is None:
                    text_res = self.text_classifier.analyze_text(text)
                self.latest_text_emotion = text_res
                self.transcript_history.append({
                    "text": text,
                    "timestamp": time.time(),
                    "dominant_emotion": text_res.dominant_emotion if text_res else "neutral",
                    "confidence": text_res.confidence if text_res else 0.0,
                })

    def set_live_text(self, text: Optional[str]):
        """Injects active text/spoken prompt into the fusion pipeline."""
        with self.lock:
            self.live_text_prompt = text
            if text and text.strip():
                self.latest_text_emotion = self.text_classifier.analyze_text(text.strip())
            elif not self.latest_transcript:
                self.latest_text_emotion = None

    def get_latest_state(self) -> Optional[MultimodalEmotionState]:
        with self.lock:
            return self.latest_state

    def get_latest_vision(self) -> Optional[VisionEmotionResult]:
        with self.lock:
            return self.latest_vision

    def get_latest_voice(self) -> Optional[VoiceEmotionResult]:
        with self.lock:
            return self.latest_voice

    def get_latest_acoustics(self) -> Optional[AcousticFeatures]:
        with self.lock:
            return self.latest_acoustics

    def get_live_text(self) -> Optional[str]:
        with self.lock:
            return self.live_text_prompt or self.latest_transcript

    def get_latest_transcript(self) -> Optional[str]:
        with self.lock:
            return self.latest_transcript

    def get_latest_text_emotion(self) -> Optional[TextEmotionResult]:
        with self.lock:
            return self.latest_text_emotion

    def get_transcript_history(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.transcript_history)


class MultimodalAudioProcessor(AudioProcessorBase):
    """Processes real-time WebRTC microphone audio buffers.

    Resamples incoming audio frames into 16kHz mono float32, maintains rolling buffers,
    calculates continuous acoustic prosody, and autonomously transcribes speech segments.
    """

    def __init__(
        self,
        context: Optional[MultimodalStreamContext] = None,
        sample_rate: int = 16000,
        enable_speech_recognition: bool = True,
    ):
        self.context = context or MultimodalStreamContext()
        self.sample_rate = sample_rate
        self.enable_speech_recognition = enable_speech_recognition
        self.audio_extractor = AcousticProsodyExtractor(sample_rate=self.sample_rate)
        self.voice_classifier = VoiceSentimentClassifier()
        self.speech_transcriber = LiveSpeechTranscriber()

        # Rolling circular buffer for 1.5 seconds of audio (24,000 samples at 16kHz)
        self.buffer_size = int(self.sample_rate * 1.5)
        self.audio_buffer = collections.deque(maxlen=self.buffer_size)

        # Speech chunk accumulator for transcription (holds up to 3.5 seconds)
        self.speech_buffer = collections.deque(maxlen=int(self.sample_rate * 3.5))
        self._silence_count = 0
        self._is_transcribing = False
        self.resampler = None
        self._step_counter = 0

    def _transcribe_worker(self, audio_array: np.ndarray):
        """Asynchronous worker executing speech-to-text without blocking the audio stream."""
        try:
            voice_res = self.context.get_latest_voice()
            res = self.speech_transcriber.transcribe_audio_array(
                audio_array,
                sample_rate=self.sample_rate,
                voice_result=voice_res,
            )
            if res and res.full_transcript:
                self.context.update_transcript(res.full_transcript, res.text_emotion)
        except Exception:
            pass
        finally:
            self._is_transcribing = False

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
                    if self.enable_speech_recognition:
                        self.speech_buffer.extend(samples_1d)

            self._step_counter += 1
            # Run acoustic feature extraction every ~4 audio frames (approx 80-120ms)
            if self._step_counter % 4 == 0 and len(self.audio_buffer) >= int(self.sample_rate * 0.25):
                buf_array = np.array(self.audio_buffer, dtype=np.float32)
                acoustics = self.audio_extractor.extract_features(buf_array)
                voice_res = self.voice_classifier.classify_voice_emotion(acoustics)
                self.context.update_voice(voice_res, acoustics)

                # Trigger speech transcription when voice pause is detected or buffer full
                if self.enable_speech_recognition and not self._is_transcribing:
                    min_samples = int(self.sample_rate * 0.5)
                    if not acoustics.speech_active:
                        self._silence_count += 1
                        if self._silence_count >= 3 and len(self.speech_buffer) >= min_samples:
                            chunk_to_transcribe = np.array(self.speech_buffer, dtype=np.float32)
                            self.speech_buffer.clear()
                            self._silence_count = 0
                            self._is_transcribing = True
                            threading.Thread(
                                target=self._transcribe_worker,
                                args=(chunk_to_transcribe,),
                                daemon=True
                            ).start()
                    else:
                        self._silence_count = 0
                        # Auto-trigger if speaker talks continuously for > 3.0 seconds
                        if len(self.speech_buffer) >= int(self.sample_rate * 3.0):
                            chunk_to_transcribe = np.array(self.speech_buffer, dtype=np.float32)
                            self.speech_buffer.clear()
                            self._is_transcribing = True
                            threading.Thread(
                                target=self._transcribe_worker,
                                args=(chunk_to_transcribe,),
                                daemon=True
                            ).start()

        except Exception:
            pass

        return frame


class MultimodalVideoProcessor(VideoTransformerBase):
    """Processes real-time video frames, overlays 468-point 3D face mesh, and renders live affective HUD."""

    def __init__(self, context: Optional[MultimodalStreamContext] = None, multi_face_mode: bool = False):
        self.context = context or MultimodalStreamContext()
        self.multi_face_mode = multi_face_mode
        self.face_mesh = FaceMeshDetector(max_num_faces=2 if multi_face_mode else 1)
        self.emotion_classifier = FacialEmotionClassifier()
        self.multi_tracker = MultiFaceTracker(max_faces=2, detector=self.face_mesh, classifier=self.emotion_classifier)
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

        # 1. Process facial tracking
        if self.multi_face_mode:
            multi_res = self.multi_tracker.track(img)
            self.context.update_multi_face(multi_res)
            img = self.multi_tracker.draw_overlay(img, multi_res)
            if multi_res.faces:
                vision_res = multi_res.faces[0].vision_result
            else:
                vision_res = VisionEmotionResult(face_detected=False)
        else:
            landmarks, head_pose = self.face_mesh.process_frame(img)
            vision_res = self.emotion_classifier.classify_emotion(landmarks, head_pose)

            if landmarks is not None:
                img = self.face_mesh.draw_mesh_overlay(img, landmarks)

                # Draw Dominant Affect HUD Card on top-left
                emo = vision_res.dominant_emotion.upper()
                conf_pct = int(vision_res.confidence * 100)

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

        # 2. Retrieve concurrent acoustic telemetry from shared stream context
        voice_res = self.context.get_latest_voice()
        acoustics = self.context.get_latest_acoustics()

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

        # 5. Render Live Subtitle Bar if speech transcript or prompt is active
        active_text = self.context.get_live_text()
        text_res = self.context.get_latest_text_emotion()
        if active_text:
            bar_y1 = max(h - 52, 10)
            bar_y2 = max(h - 14, bar_y1 + 38)
            cv2.rectangle(img, (20, bar_y1), (w - 20, bar_y2), (15, 23, 42), -1)

            accent_color = (6, 182, 212)
            if text_res and text_res.dominant_emotion == "joy":
                accent_color = (16, 185, 129)
            elif text_res and text_res.dominant_emotion in ["anger", "fear"]:
                accent_color = (239, 68, 68)
            cv2.rectangle(img, (20, bar_y1), (w - 20, bar_y2), accent_color, 1)

            sub_text = active_text[:48] + ("..." if len(active_text) > 48 else "")
            emo_tag = f" [{text_res.dominant_emotion.upper()}]" if text_res else ""
            cv2.putText(
                img, f"SPEECH: \"{sub_text}\"{emo_tag}", (30, bar_y2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (248, 250, 252), 1, cv2.LINE_AA
            )

        # 6. Tri-Modal late fusion (Vision + Audio Prosody + Spoken Text)
        fused_state = self.fusion_engine.fuse(vision=vision_res, voice=voice_res, text=text_res)
        self.context.update_state(fused_state, vision_res)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_latest_state(self) -> Optional[MultimodalEmotionState]:
        """Provides backwards compatibility for callers accessing get_latest_state directly."""
        return self.context.get_latest_state()
