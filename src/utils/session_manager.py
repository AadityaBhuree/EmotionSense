"""Session recording, timeline scrubbing, persistence, and intelligence report generator."""

import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import SESSIONS_DIR
from src.core.types import MultimodalEmotionState, SessionRecord
from src.core.config import EMOTION_LABELS


class SessionManager:
    """Manages active session state history, file persistence (JSON/CSV), and analytics summarization."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.start_time = time.time()
        self.samples: List[MultimodalEmotionState] = []
        self.is_recording = False

    def start_recording(self):
        self.samples.clear()
        self.start_time = time.time()
        self.is_recording = True

    def stop_recording(self) -> SessionRecord:
        self.is_recording = False
        return self.generate_summary()

    def add_sample(self, state: MultimodalEmotionState):
        if self.is_recording:
            self.samples.append(state)

    def generate_summary(self) -> SessionRecord:
        """Generates comprehensive session record and statistical breakdown."""
        if not self.samples:
            return SessionRecord(
                session_id=self.session_id,
                start_time=self.start_time,
                end_time=time.time(),
                samples_count=0
            )

        end_time = time.time()
        count = len(self.samples)

        # Average Affect Dimensions
        avg_v = float(sum(s.affect.valence for s in self.samples) / count)
        avg_a = float(sum(s.affect.arousal for s in self.samples) / count)
        avg_d = float(sum(s.affect.dominance for s in self.samples) / count)

        # Emotion distribution
        counts = {emo: 0 for emo in EMOTION_LABELS}
        for s in self.samples:
            counts[s.dominant_emotion] = counts.get(s.dominant_emotion, 0) + 1
        distribution = {k: round(v / count, 3) for k, v in counts.items()}

        # Average behavioral scores
        avg_eng = float(sum(s.engagement_index for s in self.samples) / count)
        avg_fat = float(sum(s.fatigue_level for s in self.samples) / count)
        avg_att = float(sum(s.attention_score for s in self.samples) / count)

        # Timeline data points
        timeline = [s.to_dict() for s in self.samples]

        # Identify key affective pivot moments (highest arousal / valence shifts)
        key_moments = []
        for i, s in enumerate(self.samples):
            if s.affect.arousal > 0.65 or abs(s.affect.valence) > 0.7:
                key_moments.append({
                    "sample_index": i,
                    "relative_time_sec": round(s.timestamp - self.start_time, 1),
                    "dominant_emotion": s.dominant_emotion,
                    "confidence": round(s.confidence, 2),
                    "valence": round(s.affect.valence, 2),
                    "arousal": round(s.affect.arousal, 2),
                })

        return SessionRecord(
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=end_time,
            samples_count=count,
            timeline=timeline,
            average_affect={"valence": round(avg_v, 3), "arousal": round(avg_a, 3), "dominance": round(avg_d, 3)},
            dominant_emotion_distribution=distribution,
            average_engagement=round(avg_eng, 3),
            average_fatigue=round(avg_fat, 3),
            average_attention=round(avg_att, 3),
            key_moments=key_moments[:10],
        )

    def save_session(self) -> Path:
        """Saves session summary to JSON in sessions directory."""
        summary = self.generate_summary()
        file_path = SESSIONS_DIR / f"{self.session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        return file_path

    def to_dataframe(self) -> pd.DataFrame:
        """Flattens session samples into a pandas DataFrame for CSV export."""
        rows = []
        for s in self.samples:
            row = {
                "timestamp": s.timestamp,
                "dominant_emotion": s.dominant_emotion,
                "confidence": s.confidence,
                "valence": s.affect.valence,
                "arousal": s.affect.arousal,
                "dominance": s.affect.dominance,
                "engagement_index": s.engagement_index,
                "fatigue_level": s.fatigue_level,
                "attention_score": s.attention_score,
            }
            # Add emotion probabilities
            for emo, p in s.probabilities.items():
                row[f"prob_{emo}"] = p
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def list_saved_sessions() -> List[Path]:
        """Lists all existing session JSON files."""
        return sorted(list(SESSIONS_DIR.glob("*.json")), reverse=True)

    @staticmethod
    def load_session(file_path: Path) -> Dict[str, Any]:
        """Loads and returns saved session JSON dictionary."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
