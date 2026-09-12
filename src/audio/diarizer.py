"""Acoustic Speaker Diarization and Conversational Turn-Taking Segmentation Engine.

Partitions multi-speaker audio recordings into distinct speaker turns, calculating per-speaker
speaking duration, acoustic prosodic profiles, conversational dominance, and interruption patterns.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import io
from pathlib import Path
import numpy as np

from src.core.types import DiarizationResult, SpeakerTurn
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.utils.logger import get_logger

logger = get_logger("AcousticDiarizer")


class AcousticDiarizer:
    """Acoustic speaker diarization and turn-taking segmentation engine."""

    def __init__(
        self,
        sample_rate: int = 16000,
        num_speakers: int = 2,
        min_speech_duration: float = 0.4,
        min_silence_duration: float = 0.35,
        energy_threshold: float = 0.015,
        prosody_extractor: Optional[AcousticProsodyExtractor] = None,
        voice_classifier: Optional[VoiceSentimentClassifier] = None,
    ):
        """Initializes the acoustic speaker diarizer.

        Args:
            sample_rate: Standard sampling rate in Hz.
            num_speakers: Expected number of speakers (default 2 for dyadic interactions).
            min_speech_duration: Minimum voiced duration in seconds to form a valid turn segment.
            min_silence_duration: Minimum silence in seconds required to split contiguous speech.
            energy_threshold: Minimum RMS energy to qualify as voiced speech.
            prosody_extractor: Optional AcousticProsodyExtractor instance.
            voice_classifier: Optional VoiceSentimentClassifier instance.
        """
        self.sample_rate = sample_rate
        self.num_speakers = max(1, num_speakers)
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.energy_threshold = energy_threshold

        self.prosody_extractor = prosody_extractor or AcousticProsodyExtractor(sample_rate=sample_rate)
        self.voice_classifier = voice_classifier or VoiceSentimentClassifier()

    def diarize_file(self, file_source: Union[str, Path, bytes, io.BytesIO]) -> DiarizationResult:
        """Loads an audio file and performs speaker diarization."""
        try:
            import soundfile as sf
            if isinstance(file_source, (str, Path)):
                audio_data, sr = sf.read(str(file_source))
            elif isinstance(file_source, bytes):
                audio_data, sr = sf.read(io.BytesIO(file_source))
            elif hasattr(file_source, "read"):
                audio_data, sr = sf.read(file_source)
            else:
                return DiarizationResult()

            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            if sr != self.sample_rate:
                try:
                    import librosa
                    audio_data = librosa.resample(
                        audio_data.astype(np.float32), orig_sr=sr, target_sr=self.sample_rate
                    )
                except Exception:
                    pass

            return self.diarize(audio_data)
        except Exception as exc:
            logger.error(f"Failed to read audio file for diarization: {exc}")
            return DiarizationResult()

    def diarize(self, audio_data: np.ndarray) -> DiarizationResult:
        """Segments a 1D float32 audio array into speaker turns with acoustic metrics."""
        if audio_data is None or len(audio_data) < int(self.sample_rate * self.min_speech_duration):
            return DiarizationResult()

        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        audio = audio_data.astype(np.float32)

        # Normalize audio peak
        max_amp = np.max(np.abs(audio))
        if max_amp > 1.0:
            audio = audio / max_amp

        total_audio_duration = float(len(audio) / self.sample_rate)

        # 1. Voice Activity Detection (VAD) Segmentation
        raw_intervals = self._detect_voice_activity(audio)
        if not raw_intervals:
            return DiarizationResult(total_audio_duration=total_audio_duration)

        # 2. Acoustic Feature Extraction for each speech interval
        features: List[np.ndarray] = []
        valid_intervals: List[Tuple[float, float]] = []

        for start_sec, end_sec in raw_intervals:
            start_idx = int(start_sec * self.sample_rate)
            end_idx = int(end_sec * self.sample_rate)
            segment = audio[start_idx:end_idx]

            feat_vec = self._extract_speaker_features(segment)
            if feat_vec is not None:
                features.append(feat_vec)
                valid_intervals.append((start_sec, end_sec))

        if not valid_intervals:
            return DiarizationResult(total_audio_duration=total_audio_duration)

        # 3. Cluster speech intervals into speaker IDs
        X = np.array(features)
        labels = self._cluster_speakers(X, min(self.num_speakers, len(valid_intervals)))

        # 4. Synthesize turns: Merge consecutive segments belonging to the same speaker
        merged_turns = self._merge_consecutive_turns(audio, valid_intervals, labels)

        # 5. Calculate global conversational dominance and durations
        speakers = sorted(list({t.speaker_id for t in merged_turns}))
        speaker_durations: Dict[str, float] = {spk: 0.0 for spk in speakers}
        for t in merged_turns:
            speaker_durations[t.speaker_id] += t.duration

        total_speech_duration = sum(speaker_durations.values())
        dominance_ratios: Dict[str, float] = {}
        for spk, dur in speaker_durations.items():
            dominance_ratios[spk] = (
                round(dur / total_speech_duration, 3) if total_speech_duration > 0 else 0.0
            )

        # 6. Count conversational interruptions / rapid overlaps
        interruption_count = self._count_interruptions(merged_turns)

        return DiarizationResult(
            turns=merged_turns,
            speakers=speakers,
            speaker_durations={k: round(v, 2) for k, v in speaker_durations.items()},
            dominance_ratios=dominance_ratios,
            interruption_count=interruption_count,
            total_speech_duration=round(total_speech_duration, 2),
            total_audio_duration=round(total_audio_duration, 2),
        )

    def _detect_voice_activity(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Segments continuous audio into voiced intervals with hangover smoothing."""
        frame_len = int(self.sample_rate * 0.05)  # 50ms window
        hop_len = int(self.sample_rate * 0.025)   # 25ms hop
        if len(audio) < frame_len:
            return []

        # Calculate short-time RMS energy per frame
        n_frames = (len(audio) - frame_len) // hop_len + 1
        energies = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            idx = i * hop_len
            frame = audio[idx : idx + frame_len]
            energies[i] = np.sqrt(np.mean(frame**2))

        # Dynamic energy thresholding
        avg_energy = float(np.mean(energies))
        thresh = max(self.energy_threshold, avg_energy * 0.45)
        voiced_mask = energies > thresh

        # Extract voiced frame ranges
        intervals: List[Tuple[int, int]] = []
        in_speech = False
        start_frame = 0

        for i, is_voiced in enumerate(voiced_mask):
            if is_voiced and not in_speech:
                in_speech = True
                start_frame = i
            elif not is_voiced and in_speech:
                in_speech = False
                intervals.append((start_frame, i))
        if in_speech:
            intervals.append((start_frame, len(voiced_mask)))

        if not intervals:
            return []

        # Hangover smoothing: merge intervals separated by short silence (< min_silence_duration)
        silence_frame_thresh = int(self.min_silence_duration / 0.025)
        merged_intervals: List[Tuple[int, int]] = [intervals[0]]

        for start_f, end_f in intervals[1:]:
            prev_start, prev_end = merged_intervals[-1]
            gap = start_f - prev_end
            if gap <= silence_frame_thresh:
                merged_intervals[-1] = (prev_start, end_f)
            else:
                merged_intervals.append((start_f, end_f))

        # Convert frame indices to seconds, filtering intervals < min_speech_duration
        min_speech_frames = int(self.min_speech_duration / 0.025)
        final_intervals: List[Tuple[float, float]] = []

        for start_f, end_f in merged_intervals:
            if (end_f - start_f) >= min_speech_frames:
                start_sec = round(start_f * 0.025, 2)
                end_sec = round(min(len(audio) / self.sample_rate, end_f * 0.025 + 0.05), 2)
                final_intervals.append((start_sec, end_sec))

        return final_intervals

    def _extract_speaker_features(self, segment: np.ndarray) -> Optional[np.ndarray]:
        """Extracts acoustic embedding vector for speaker clustering."""
        if len(segment) < int(self.sample_rate * 0.2):
            return None

        features: List[float] = []

        # 1. Pitch characteristics (F0 mean, F0 std, confidence)
        pro = self.prosody_extractor.extract_features(segment)
        features.extend([
            pro.pitch_hz,
            pro.pitch_confidence,
            pro.rms_energy,
            pro.jitter_percent,
            pro.shimmer_percent,
            pro.zero_crossing_rate,
        ])

        # 2. Spectral centroid & bandwidth via librosa or FFT fallback
        try:
            import librosa
            centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sample_rate)
            rolloff = librosa.feature.spectral_rolloff(y=segment, sr=self.sample_rate)
            mfccs = librosa.feature.mfcc(y=segment, sr=self.sample_rate, n_mfcc=13)

            features.append(float(np.mean(centroid)))
            features.append(float(np.std(centroid)))
            features.append(float(np.mean(rolloff)))

            # Mean & std of top 8 MFCCs
            for i in range(min(8, mfccs.shape[0])):
                features.append(float(np.mean(mfccs[i])))
                features.append(float(np.std(mfccs[i])))
        except Exception:
            # Resilient FFT-based spectral energy bins fallback
            fft_vals = np.abs(np.fft.rfft(segment))
            features.append(float(np.mean(fft_vals[: len(fft_vals) // 4])))
            features.append(float(np.mean(fft_vals[len(fft_vals) // 4 :])))

        vec = np.array(features, dtype=np.float32)
        # Replace NaN or Inf
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
        return vec

    def _cluster_speakers(self, X: np.ndarray, k: int) -> List[int]:
        """Clusters acoustic feature vectors into speaker IDs."""
        if len(X) <= 1 or k <= 1:
            return [0] * len(X)

        # Standardize features
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-6
        X_norm = (X - mean) / std

        # Try sklearn KMeans first
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_norm)
            return [int(lbl) for lbl in labels]
        except Exception:
            pass

        # Resilient Pure-NumPy K-Means implementation
        return self._pure_numpy_kmeans(X_norm, k)

    def _pure_numpy_kmeans(self, X: np.ndarray, k: int, max_iter: int = 50) -> List[int]:
        """Robust pure-NumPy K-Means clustering algorithm."""
        n_samples = X.shape[0]
        if n_samples <= k:
            return list(range(n_samples))

        # K-Means++ initialization
        centroids = [X[0]]
        for _ in range(1, k):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            next_idx = int(np.argmax(dists))
            centroids.append(X[next_idx])
        centroids = np.array(centroids)

        labels = np.zeros(n_samples, dtype=int)
        for _ in range(max_iter):
            # Assign points to nearest centroid
            dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # Recompute centroids
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    centroids[j] = np.mean(X[mask], axis=0)

        return [int(lbl) for lbl in labels]

    def _merge_consecutive_turns(
        self,
        audio: np.ndarray,
        intervals: List[Tuple[float, float]],
        labels: List[int],
    ) -> List[SpeakerTurn]:
        """Merges consecutive intervals from the same speaker and computes turn-level affect."""
        if not intervals:
            return []

        # Assign speaker labels ("Speaker_0", "Speaker_1", etc.)
        speaker_labels = [f"Speaker_{lbl}" for lbl in labels]

        merged: List[Dict[str, Any]] = []
        cur_speaker = speaker_labels[0]
        cur_start, cur_end = intervals[0]

        for i in range(1, len(intervals)):
            spk = speaker_labels[i]
            st, et = intervals[i]

            # If same speaker and gap is under 0.8s, merge turns
            if spk == cur_speaker and (st - cur_end) <= 0.8:
                cur_end = et
            else:
                merged.append({"speaker": cur_speaker, "start": cur_start, "end": cur_end})
                cur_speaker = spk
                cur_start = st
                cur_end = et

        merged.append({"speaker": cur_speaker, "start": cur_start, "end": cur_end})

        # Build SpeakerTurn instances with acoustic metrics
        turns: List[SpeakerTurn] = []
        for item in merged:
            start_t = item["start"]
            end_t = item["end"]
            dur = round(end_t - start_t, 2)

            start_sample = int(start_t * self.sample_rate)
            end_sample = int(end_t * self.sample_rate)
            chunk = audio[start_sample:end_sample]

            prosody = self.prosody_extractor.extract_features(chunk)
            voice_res = self.voice_classifier.classify(prosody)

            turns.append(
                SpeakerTurn(
                    speaker_id=item["speaker"],
                    start_time=start_t,
                    end_time=end_t,
                    duration=dur,
                    transcript="",
                    acoustics=prosody,
                    affect=voice_res.affect,
                    dominant_emotion=voice_res.dominant_emotion,
                )
            )

        return turns

    def _count_interruptions(self, turns: List[SpeakerTurn]) -> int:
        """Calculates instances where turn transitions exhibit negative latency (overlap)."""
        if len(turns) < 2:
            return 0

        interruptions = 0
        for i in range(1, len(turns)):
            prev_turn = turns[i - 1]
            cur_turn = turns[i]

            # Transition between different speakers
            if cur_turn.speaker_id != prev_turn.speaker_id:
                # If current turn starts before previous turn ends (overlap)
                # or starts within 0.05s with elevated pitch/energy
                if cur_turn.start_time < prev_turn.end_time:
                    interruptions += 1
                elif (cur_turn.start_time - prev_turn.end_time) < 0.08:
                    cur_energy = cur_turn.acoustics.rms_energy if cur_turn.acoustics else 0.0
                    prev_energy = prev_turn.acoustics.rms_energy if prev_turn.acoustics else 0.0
                    if cur_energy > prev_energy * 1.35:
                        interruptions += 1

        return interruptions
