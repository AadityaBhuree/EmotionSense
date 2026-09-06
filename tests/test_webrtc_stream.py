"""Unit and integration tests for real-time WebRTC audio and video stream processors."""

import pytest
import numpy as np
import av
import threading
import time

from src.ui.video_processor import (
    MultimodalStreamContext,
    MultimodalAudioProcessor,
    MultimodalVideoProcessor,
)
from src.core.types import AcousticFeatures, VoiceEmotionResult, VisionEmotionResult, MultimodalEmotionState


def test_stream_context_initialization():
    ctx = MultimodalStreamContext(window_size=15)
    assert ctx.get_latest_state() is None
    assert ctx.get_latest_voice() is None
    assert ctx.get_latest_acoustics() is None
    assert ctx.speech_active is False
    assert ctx.sample_count == 0


def test_stream_context_thread_safe_updates():
    ctx = MultimodalStreamContext()

    def worker_audio():
        for i in range(10):
            acoustics = AcousticFeatures(pitch_hz=220.0 + i, rms_energy=0.08, speech_active=True)
            voice_res = VoiceEmotionResult(dominant_emotion="joy", confidence=0.85, acoustics=acoustics)
            ctx.update_voice(voice_res, acoustics)
            time.sleep(0.005)

    def worker_video():
        for i in range(10):
            vision = VisionEmotionResult(face_detected=True, dominant_emotion="joy", confidence=0.9)
            state = MultimodalEmotionState(dominant_emotion="joy", confidence=0.88)
            ctx.update_state(state, vision)
            time.sleep(0.005)

    t1 = threading.Thread(target=worker_audio)
    t2 = threading.Thread(target=worker_video)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert ctx.sample_count == 10
    assert ctx.get_latest_voice() is not None
    assert ctx.get_latest_voice().dominant_emotion == "joy"
    assert ctx.get_latest_state() is not None
    assert ctx.speech_active is True


def test_multimodal_audio_processor_synthetic_audio():
    ctx = MultimodalStreamContext()
    audio_proc = MultimodalAudioProcessor(context=ctx, sample_rate=16000)

    # Generate synthetic 220 Hz tone frame: 48kHz stereo, 16-bit PCM (well within human vocal range 65-400 Hz)
    sr = 48000
    duration = 0.1  # 100ms per frame
    target_f0 = 220.0
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * target_f0 * t) * 32767).astype(np.int16)
    stereo_pcm = np.stack([sine, sine], axis=1)  # shape (num_samples, 2)

    # Feed multiple frames to accumulate >= 0.25s of audio
    for _ in range(5):
        frame = av.AudioFrame(format="s16", layout="stereo", samples=num_samples)
        frame.rate = sr
        frame.planes[0].update(stereo_pcm.tobytes())
        ret_frame = audio_proc.recv(frame)
        assert ret_frame is frame

    # Verify that acoustic features were extracted and pushed to context
    latest_voice = ctx.get_latest_voice()
    latest_acoustics = ctx.get_latest_acoustics()

    assert latest_voice is not None
    assert latest_acoustics is not None
    assert latest_acoustics.speech_active is True
    assert latest_acoustics.rms_energy > 0.01
    # Pitch should be within 15 Hz of 220 Hz
    assert abs(latest_acoustics.pitch_hz - target_f0) < 15.0


def test_multimodal_video_processor_synthetic_frame():
    ctx = MultimodalStreamContext()
    video_proc = MultimodalVideoProcessor(context=ctx)

    # Provide active acoustic context
    acoustics = AcousticFeatures(pitch_hz=210.0, rms_energy=0.06, speech_active=True)
    voice_res = VoiceEmotionResult(dominant_emotion="joy", confidence=0.82, acoustics=acoustics)
    ctx.update_voice(voice_res, acoustics)
    ctx.set_live_text("Hello EmotionSense!")

    # Create synthetic video frame (480x640 BGR)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    frame = av.VideoFrame.from_ndarray(canvas, format="bgr24")

    # Process frame
    out_frame = video_proc.recv(frame)
    assert isinstance(out_frame, av.VideoFrame)

    out_arr = out_frame.to_ndarray(format="bgr24")
    assert out_arr.shape == (480, 640, 3)

    # State should have been fused and saved in context
    latest_state = ctx.get_latest_state()
    assert latest_state is not None
    assert isinstance(latest_state, MultimodalEmotionState)
    assert ctx.sample_count == 1
