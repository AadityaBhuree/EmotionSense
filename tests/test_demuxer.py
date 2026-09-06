"""Unit and integration tests for AudiovisualDemuxer and synchronized multimodal fusion."""

import io
import pytest
import numpy as np
import av

from src.utils.demuxer import AudiovisualDemuxer, DemuxResult
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.core.config import EMOTION_LABELS


def create_synthetic_mp4_container(
    has_audio: bool = True,
    duration_sec: float = 1.0,
    video_fps: int = 25,
    sample_rate: int = 16000,
    audio_freq: float = 440.0,
) -> io.BytesIO:
    """Helper function generating an in-memory MP4 container with video and optional audio."""
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        # Video stream
        v_stream = container.add_stream("mpeg4", rate=video_fps)
        v_stream.width = 64
        v_stream.height = 64
        v_stream.pix_fmt = "yuv420p"

        # Audio stream
        a_stream = None
        if has_audio:
            a_stream = container.add_stream("aac", rate=sample_rate)
            a_stream.layout = "mono"

        # Write video frames
        num_frames = max(1, int(duration_sec * video_fps))
        for _ in range(num_frames):
            frame = av.VideoFrame(64, 64, "rgb24")
            for packet in v_stream.encode(frame):
                container.mux(packet)

        # Write audio frames
        if has_audio and a_stream is not None:
            total_samples = int(duration_sec * sample_rate)
            t = np.linspace(0, duration_sec, total_samples, endpoint=False)
            sine_wave = (0.5 * np.sin(2 * np.pi * audio_freq * t)).astype(np.float32)

            chunk_size = 1024
            for i in range(0, total_samples, chunk_size):
                chunk = sine_wave[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                # 16-bit PCM conversion for AAC encoding
                pcm16 = (chunk * 32767.0).astype(np.int16)
                a_frame = av.AudioFrame(format="s16p", layout="mono", samples=chunk_size)
                a_frame.sample_rate = sample_rate
                a_frame.planes[0].update(pcm16.tobytes())

                for packet in a_stream.encode(a_frame):
                    container.mux(packet)

        # Flush streams
        for packet in v_stream.encode():
            container.mux(packet)
        if has_audio and a_stream is not None:
            for packet in a_stream.encode():
                container.mux(packet)

    buf.seek(0)
    return buf


def test_demuxer_initialization():
    demuxer = AudiovisualDemuxer(target_sample_rate=22050)
    assert demuxer.target_sample_rate == 22050


def test_demuxer_synthetic_video_with_audio():
    demuxer = AudiovisualDemuxer(target_sample_rate=16000)
    buf = create_synthetic_mp4_container(has_audio=True, duration_sec=1.5, audio_freq=300.0)

    result = demuxer.demux(buf)

    assert isinstance(result, DemuxResult)
    assert result.has_audio is True
    assert result.sample_rate == 16000
    assert result.channels == 1
    assert len(result.audio_array) > 0
    assert result.duration_seconds > 0.5
    assert result.video_fps > 0
    assert np.max(np.abs(result.audio_array)) <= 1.0


def test_demuxer_synthetic_silent_video():
    demuxer = AudiovisualDemuxer(target_sample_rate=16000)
    buf = create_synthetic_mp4_container(has_audio=False, duration_sec=1.0)

    result = demuxer.demux(buf)

    assert isinstance(result, DemuxResult)
    assert result.has_audio is False
    assert len(result.audio_array) == 0
    assert result.duration_seconds == 0.0
    assert result.channels == 0


def test_demuxer_corrupted_and_missing_input():
    demuxer = AudiovisualDemuxer(target_sample_rate=16000)

    # Corrupt bytes
    corrupt_result = demuxer.demux(b"INVALID_CORRUPTED_VIDEO_BYTES")
    assert corrupt_result.has_audio is False
    assert len(corrupt_result.audio_array) == 0

    # Non-existent file path
    missing_result = demuxer.demux("path/does/not/exist.mp4")
    assert missing_result.has_audio is False
    assert "error" in missing_result.metadata


def test_audio_window_extraction():
    demuxer = AudiovisualDemuxer(target_sample_rate=16000)
    # 2 seconds of synthetic audio
    audio_data = np.ones(32000, dtype=np.float32)

    # 1. Centered window around t=1.0s (should be full ones)
    w_mid = demuxer.get_audio_window(audio_data, timestamp_sec=1.0, window_sec=1.0)
    assert len(w_mid) == 16000
    assert np.allclose(w_mid, 1.0)

    # 2. Window centered at t=0.0s (start before 0, padded with zeros in prefix)
    w_start = demuxer.get_audio_window(audio_data, timestamp_sec=0.0, window_sec=1.0, centered=True)
    assert len(w_start) == 16000
    # First half should be 0, second half 1.0
    assert np.allclose(w_start[:8000], 0.0)
    assert np.allclose(w_start[8000:], 1.0)

    # 3. Trailing window at t=1.0s (window covers [0.0, 1.0])
    w_trail = demuxer.get_audio_window(audio_data, timestamp_sec=1.0, window_sec=1.0, centered=False)
    assert len(w_trail) == 16000
    assert np.allclose(w_trail, 1.0)

    # 4. Empty audio handling
    w_empty = demuxer.get_audio_window(np.array([], dtype=np.float32), timestamp_sec=1.0, window_sec=1.0)
    assert len(w_empty) == 16000
    assert np.allclose(w_empty, 0.0)


def test_generate_synchronized_timeline():
    demuxer = AudiovisualDemuxer(target_sample_rate=16000)
    audio_data = np.random.uniform(-0.5, 0.5, 32000).astype(np.float32)

    timestamps = [0.0, 0.5, 1.0, 1.5]
    windows = demuxer.generate_synchronized_timeline(audio_data, timestamps, window_sec=0.5)

    assert len(windows) == len(timestamps)
    for w in windows:
        assert len(w) == int(0.5 * 16000)


def test_synchronized_audiovisual_late_fusion():
    """End-to-end integration: Demux synthetic container and run late fusion across both channels."""
    demuxer = AudiovisualDemuxer(target_sample_rate=16000)
    prosody_extractor = AcousticProsodyExtractor(sample_rate=16000)
    voice_classifier = VoiceSentimentClassifier()
    face_classifier = FacialEmotionClassifier()
    fusion_engine = MultimodalFusionEngine()

    # Create 1.5s video with 220Hz audio tone
    buf = create_synthetic_mp4_container(has_audio=True, duration_sec=1.5, audio_freq=220.0)
    demux_res = demuxer.demux(buf)

    assert demux_res.has_audio is True

    # Checkpoint at timestamp 0.5s
    audio_chunk = demuxer.get_audio_window(demux_res.audio_array, timestamp_sec=0.5, window_sec=1.0)
    acoustics = prosody_extractor.extract_features(audio_chunk)
    voice_res = voice_classifier.classify_voice_emotion(acoustics)

    # Synthetic dummy video frame (blank 128x128 image)
    dummy_frame = np.zeros((128, 128, 3), dtype=np.uint8)
    vision_res = face_classifier.classify(dummy_frame)

    fused_state = fusion_engine.fuse(vision=vision_res, voice=voice_res)

    assert fused_state is not None
    assert fused_state.dominant_emotion in EMOTION_LABELS
    assert fused_state.voice is not None
    assert fused_state.voice.acoustics.speech_active is True
    assert 0.0 <= fused_state.confidence <= 1.0
    assert abs(sum(fused_state.probabilities.values()) - 1.0) < 1e-4
