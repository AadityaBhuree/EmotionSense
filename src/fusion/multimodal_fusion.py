"""Temporal late fusion engine combining visual, acoustic, and contextual affect signals."""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque

from src.core.types import (
    VisionEmotionResult,
    VoiceEmotionResult,
    AffectVector,
    MultimodalEmotionState,
)
from src.core.config import (
    EMOTION_LABELS,
    EMOTION_VAD_COORDINATES,
    DEFAULT_MODALITY_WEIGHTS,
)
from src.fusion.metrics import AffectMetricsCalculator


class MultimodalFusionEngine:
    """Synchronizes and fuses multi-modal affective streams across a temporal sliding window."""

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.history: deque[MultimodalEmotionState] = deque(maxlen=window_size)
        self.metrics_calc = AffectMetricsCalculator()
        self.weights = DEFAULT_MODALITY_WEIGHTS.copy()

    def fuse(
        self,
        vision: Optional[VisionEmotionResult],
        voice: Optional[VoiceEmotionResult] = None,
        text: Optional[Any] = None,
    ) -> MultimodalEmotionState:
        """Fuses vision, audio, and text streams into a unified MultimodalEmotionState."""
        current_time = time.time()

        # 1. Determine modality presence & dynamic weighting
        has_vision = vision is not None and vision.face_detected
        has_voice = voice is not None and voice.acoustics.speech_active
        has_text = text is not None and bool(getattr(text, "text", ""))

        w_vis = self.weights.get("vision", 0.45) if has_vision else 0.0
        w_aud = self.weights.get("audio", 0.30) if has_voice else 0.0
        w_txt = self.weights.get("text", 0.35) if has_text else 0.0
        w_ctx = self.weights.get("context", 0.10)

        total_weight = w_vis + w_aud + w_txt + w_ctx
        if total_weight > 0:
            w_vis /= total_weight
            w_aud /= total_weight
            w_txt /= total_weight
            w_ctx /= total_weight
        else:
            w_ctx = 1.0

        # 2. Weighted late fusion of emotion probabilities
        fused_probs: Dict[str, float] = {}
        for emo in EMOTION_LABELS:
            p_vis = vision.probabilities.get(emo, 0.0) if has_vision else 0.0
            p_aud = voice.probabilities.get(emo, 0.0) if has_voice else 0.0
            p_txt = text.probabilities.get(emo, 0.0) if has_text else 0.0
            
            # Prior temporal smoothing from history
            p_hist = 0.0
            if len(self.history) > 0:
                p_hist = float(np.mean([h.probabilities.get(emo, 0.0) for h in self.history]))
            else:
                p_hist = 1.0 if emo == "neutral" else 0.0

            fused_score = (p_vis * w_vis) + (p_aud * w_aud) + (p_txt * w_txt) + (p_hist * w_ctx)
            fused_probs[emo] = float(fused_score)

        # Normalize probabilities sum to 1.0
        prob_sum = sum(fused_probs.values())
        if prob_sum > 0:
            fused_probs = {k: float(v / prob_sum) for k, v in fused_probs.items()}
        else:
            fused_probs = {emo: (1.0 if emo == "neutral" else 0.0) for emo in EMOTION_LABELS}

        dominant_emotion = max(fused_probs, key=fused_probs.get)
        confidence = fused_probs[dominant_emotion]

        # 3. Compute continuous 3D Affect Vector (Valence, Arousal, Dominance)
        valence = 0.0
        arousal = 0.0
        dominance = 0.0
        for emo, p in fused_probs.items():
            vad = EMOTION_VAD_COORDINATES.get(emo, [0.0, 0.0, 0.0])
            valence += p * vad[0]
            arousal += p * vad[1]
            dominance += p * vad[2]

        affect = AffectVector(
            valence=float(np.clip(valence, -1.0, 1.0)),
            arousal=float(np.clip(arousal, -1.0, 1.0)),
            dominance=float(np.clip(dominance, -1.0, 1.0)),
        )

        # 4. Behavioral higher-order metrics
        attention = self.metrics_calc.calculate_attention_score(vision)
        fatigue = self.metrics_calc.calculate_fatigue_level(vision, voice.acoustics if voice else None)
        engagement = self.metrics_calc.calculate_engagement_index(vision, voice, attention)

        state = MultimodalEmotionState(
            timestamp=current_time,
            dominant_emotion=dominant_emotion,
            confidence=confidence,
            probabilities=fused_probs,
            affect=affect,
            engagement_index=engagement,
            fatigue_level=fatigue,
            attention_score=attention,
            vision=vision,
            voice=voice,
            text=text,
        )

        self.history.append(state)
        return state

    def get_recent_history(self) -> List[MultimodalEmotionState]:
        return list(self.history)
