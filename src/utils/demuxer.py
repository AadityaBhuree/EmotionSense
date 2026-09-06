"""Audiovisual Demuxer and Temporal Alignment Engine for Multi-Modal Containers.

Extracts audio streams from video files (MP4, AVI, MOV, MKV, WebM) in memory or from disk,
resamples to target sampling rate (16kHz mono), and provides synchronized audio windowing
for concurrent visual-acoustic affect classification.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import io
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("AudiovisualDemuxer")

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    logger.warning("PyAV ('av') is not installed. Container demuxing will be restricted.")


@dataclass
class DemuxResult:
    """Encapsulates extracted audio data and container metadata."""

    has_audio: bool = False
    audio_array: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sample_rate: int = 16000
    duration_seconds: float = 0.0
    channels: int = 1
    video_fps: float = 30.0
    video_frame_count: int = 0
    video_duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AudiovisualDemuxer:
    """Extracts, resamples, and synchronizes audio tracks from video containers."""

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def demux(self, source: Union[str, Path, bytes, io.BytesIO]) -> DemuxResult:
        """Demuxes audio from a video file path, raw bytes, or BytesIO buffer.

        Args:
            source: Path to file, raw bytes, or in-memory BytesIO buffer.

        Returns:
            DemuxResult containing normalized float32 mono audio and stream telemetry.
        """
        if not AV_AVAILABLE:
            logger.error("PyAV is required for AudiovisualDemuxer.demux()")
            return DemuxResult(sample_rate=self.target_sample_rate)

        # Prepare container source
        container_input: Union[str, io.BytesIO]
        if isinstance(source, bytes):
            container_input = io.BytesIO(source)
        elif isinstance(source, Path):
            container_input = str(source)
        else:
            container_input = source

        try:
            container = av.open(container_input)
        except Exception as exc:
            logger.error(f"Failed to open media container: {exc}")
            return DemuxResult(sample_rate=self.target_sample_rate, metadata={"error": str(exc)})

        try:
            # 1. Video metadata extraction
            has_video = len(container.streams.video) > 0
            video_fps = 30.0
            video_frame_count = 0
            video_duration = 0.0

            if has_video:
                v_stream = container.streams.video[0]
                if v_stream.average_rate:
                    video_fps = float(v_stream.average_rate)
                elif v_stream.base_rate:
                    video_fps = float(v_stream.base_rate)
                video_frame_count = int(v_stream.frames) if v_stream.frames else 0
                if v_stream.duration and v_stream.time_base:
                    video_duration = float(v_stream.duration * v_stream.time_base)

            # 2. Audio track extraction & resampling
            has_audio = len(container.streams.audio) > 0
            if not has_audio:
                logger.info("Container contains no audio streams.")
                return DemuxResult(
                    has_audio=False,
                    audio_array=np.array([], dtype=np.float32),
                    sample_rate=self.target_sample_rate,
                    duration_seconds=0.0,
                    channels=0,
                    video_fps=video_fps,
                    video_frame_count=video_frame_count,
                    video_duration_seconds=video_duration,
                    metadata={"video_codecs": [s.codec_context.name for s in container.streams.video]},
                )

            a_stream = container.streams.audio[0]
            resampler = av.audio.resampler.AudioResampler(
                format="fltp",
                layout="mono",
                rate=self.target_sample_rate,
            )

            chunks: List[np.ndarray] = []
            for frame in container.decode(audio=0):
                resampled_frames = resampler.resample(frame)
                for rf in resampled_frames:
                    chunks.append(rf.to_ndarray())

            # Flush resampler buffer
            flushed_frames = resampler.resample(None)
            for rf in flushed_frames:
                chunks.append(rf.to_ndarray())

            if chunks:
                audio_concatenated = np.concatenate(chunks, axis=1).squeeze().astype(np.float32)
                # Handle single sample or empty edge-case
                if audio_concatenated.ndim == 0:
                    audio_array = np.array([float(audio_concatenated)], dtype=np.float32)
                else:
                    audio_array = audio_concatenated
            else:
                audio_array = np.array([], dtype=np.float32)

            duration = float(len(audio_array) / self.target_sample_rate) if len(audio_array) > 0 else 0.0

            # Normalize amplitude if needed
            max_val = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0.0
            if max_val > 1.0:
                audio_array = audio_array / max_val

            return DemuxResult(
                has_audio=True,
                audio_array=audio_array,
                sample_rate=self.target_sample_rate,
                duration_seconds=duration,
                channels=1,
                video_fps=video_fps,
                video_frame_count=video_frame_count,
                video_duration_seconds=video_duration if video_duration > 0.0 else duration,
                metadata={
                    "audio_codec": a_stream.codec_context.name if a_stream.codec_context else "unknown",
                    "original_sample_rate": a_stream.sample_rate or 0,
                    "original_channels": a_stream.channels or 0,
                },
            )

        except Exception as exc:
            logger.error(f"Error during demuxing process: {exc}")
            return DemuxResult(
                has_audio=False,
                sample_rate=self.target_sample_rate,
                metadata={"error": str(exc)},
            )
        finally:
            container.close()

    def get_audio_window(
        self,
        audio_data: np.ndarray,
        timestamp_sec: float,
        window_sec: float = 1.0,
        sample_rate: Optional[int] = None,
        centered: bool = True,
    ) -> np.ndarray:
        """Extracts a fixed-duration audio segment corresponding to a video timestamp.

        Args:
            audio_data: 1D float32 audio array.
            timestamp_sec: Video timestamp checkpoint in seconds.
            window_sec: Window length in seconds (default 1.0s).
            sample_rate: Sampling rate (defaults to target_sample_rate).
            centered: If True, window is centered at timestamp_sec; if False, window ends at timestamp_sec.

        Returns:
            1D float32 numpy array with exactly window_sec * sample_rate samples, zero-padded if out of bounds.
        """
        sr = sample_rate or self.target_sample_rate
        total_window_samples = max(1, int(window_sec * sr))

        if audio_data is None or len(audio_data) == 0:
            return np.zeros(total_window_samples, dtype=np.float32)

        if centered:
            start_sec = timestamp_sec - (window_sec / 2.0)
        else:
            start_sec = timestamp_sec - window_sec

        start_idx = int(round(start_sec * sr))
        end_idx = start_idx + total_window_samples

        out = np.zeros(total_window_samples, dtype=np.float32)

        src_start = max(0, start_idx)
        src_end = min(len(audio_data), end_idx)

        dst_start = max(0, -start_idx)
        copy_len = src_end - src_start

        if copy_len > 0 and dst_start < total_window_samples:
            actual_copy_len = min(copy_len, total_window_samples - dst_start)
            out[dst_start : dst_start + actual_copy_len] = audio_data[src_start : src_start + actual_copy_len]

        return out

    def generate_synchronized_timeline(
        self,
        audio_data: np.ndarray,
        video_timestamps: List[float],
        window_sec: float = 1.0,
        sample_rate: Optional[int] = None,
        centered: bool = True,
    ) -> List[np.ndarray]:
        """Slices synchronized audio windows for each video checkpoint timestamp."""
        return [
            self.get_audio_window(
                audio_data=audio_data,
                timestamp_sec=ts,
                window_sec=window_sec,
                sample_rate=sample_rate,
                centered=centered,
            )
            for ts in video_timestamps
        ]
