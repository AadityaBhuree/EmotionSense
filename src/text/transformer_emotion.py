"""Deep Learning Transformer Emotion Classification Backend.

Integrates HuggingFace Transformer models (e.g. RoBERTa-GoEmotions / Twitter-RoBERTa)
with PyTorch, mapping deep neural contextual representations into standard 8-Ekman
discrete categories, extended nuances, and continuous 3D VAD spaces.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.core.types import TextEmotionResult, AffectVector, TokenSalience, EmotionCategory


class TransformerEmotionClassifier:
    """Deep learning neural text emotion classifier using HuggingFace / PyTorch."""

    # Taxonomy mapping from 28 GoEmotions / RoBERTa categories to 8 Ekman categories
    GO_EMOTION_TO_EKMAN = {
        "admiration": ("joy", "love"),
        "amusement": ("joy", "excitement"),
        "anger": ("anger", "frustration"),
        "annoyance": ("anger", "frustration"),
        "approval": ("joy", "optimism"),
        "caring": ("joy", "empathy"),
        "confusion": ("surprise", "confusion"),
        "curiosity": ("surprise", "excitement"),
        "desire": ("joy", "love"),
        "disappointment": ("sadness", "frustration"),
        "disapproval": ("contempt", "frustration"),
        "disgust": ("disgust", "contempt"),
        "embarrassment": ("fear", "anxiety"),
        "excitement": ("joy", "excitement"),
        "fear": ("fear", "anxiety"),
        "gratitude": ("joy", "gratitude"),
        "grief": ("sadness", "sadness"),
        "joy": ("joy", "joy"),
        "love": ("joy", "love"),
        "nervousness": ("fear", "anxiety"),
        "optimism": ("joy", "optimism"),
        "pride": ("joy", "optimism"),
        "realization": ("surprise", "confusion"),
        "relief": ("joy", "optimism"),
        "remorse": ("sadness", "sadness"),
        "sadness": ("sadness", "sadness"),
        "surprise": ("surprise", "excitement"),
        "neutral": ("neutral", "neutral"),
    }

    # 3D VAD Base References
    VAD_BASES: Dict[str, Tuple[float, float, float]] = {
        "joy": (0.81, 0.51, 0.46),
        "sadness": (-0.63, -0.27, -0.33),
        "anger": (-0.51, 0.59, 0.25),
        "fear": (-0.64, 0.60, -0.43),
        "surprise": (0.40, 0.67, -0.13),
        "disgust": (-0.60, 0.35, 0.11),
        "neutral": (0.00, 0.00, 0.00),
        "contempt": (-0.55, 0.38, 0.45),
    }

    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions"):
        """Initializes the transformer classifier with lazy loading."""
        self.model_name = model_name
        self.pipeline = None
        self._is_loaded = False
        self._load_error: Optional[str] = None
        self.device = "cpu"

    def load_model(self) -> bool:
        """Loads the transformer model pipeline into memory."""
        if self._is_loaded and self.pipeline is not None:
            return True

        try:
            import torch
            from transformers import pipeline

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=0 if self.device == "cuda" else -1,
            )
            self._is_loaded = True
            return True
        except Exception as e:
            self._load_error = str(e)
            self._is_loaded = False
            return False

    @property
    def is_available(self) -> bool:
        """Returns True if PyTorch and Transformers can be imported."""
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False

    def predict(self, text: str) -> TextEmotionResult:
        """Runs neural transformer inference on input text."""
        if not text or not text.strip():
            return TextEmotionResult(
                text="",
                dominant_emotion="neutral",
                confidence=1.0,
                probabilities={k: (1.0 if k == "neutral" else 0.0) for k in self.VAD_BASES},
                affect=AffectVector(0.0, 0.0, 0.0),
                empathy_advice="No text content provided."
            )

        clean_text = text.strip()
        t0 = time.time()

        # If model is not loaded yet, attempt lazy load
        if not self._is_loaded:
            loaded = self.load_model()
            if not loaded:
                # Fallback to local rule simulation if offline or download error
                from src.text.nlp_emotion import TextEmotionClassifier
                fallback_clf = TextEmotionClassifier()
                res = fallback_clf.analyze_text(clean_text)
                return res

        # Run transformer forward pass
        raw_outputs = self.pipeline(clean_text[:512])[0]
        
        # Accumulate probabilities into 8-Ekman and nuances
        ekman_probs: Dict[str, float] = {k: 0.0 for k in self.VAD_BASES}
        nuances: Dict[str, float] = {}

        for item in raw_outputs:
            label = item["label"].lower()
            score = float(item["score"])

            if label in self.GO_EMOTION_TO_EKMAN:
                ekman_cat, nuance_cat = self.GO_EMOTION_TO_EKMAN[label]
                ekman_probs[ekman_cat] += score
                nuances[nuance_cat] = max(nuances.get(nuance_cat, 0.0), score)
            else:
                ekman_probs["neutral"] += score * 0.5

        # Normalize Ekman probabilities
        total_p = sum(ekman_probs.values())
        if total_p > 0:
            ekman_probs = {k: float(v / total_p) for k, v in ekman_probs.items()}
        else:
            ekman_probs = {k: (1.0 if k == "neutral" else 0.0) for k in self.VAD_BASES}

        dom_emo = max(ekman_probs, key=ekman_probs.get)
        confidence = ekman_probs[dom_emo]

        # Compute 3D VAD
        v = sum(ekman_probs[e] * self.VAD_BASES[e][0] for e in ekman_probs)
        a = sum(ekman_probs[e] * self.VAD_BASES[e][1] for e in ekman_probs)
        d = sum(ekman_probs[e] * self.VAD_BASES[e][2] for e in ekman_probs)

        affect = AffectVector(
            valence=round(float(np.clip(v, -1.0, 1.0)), 3),
            arousal=round(float(np.clip(a, -1.0, 1.0)), 3),
            dominance=round(float(np.clip(d, -1.0, 1.0)), 3),
        )

        pos_mass = ekman_probs.get("joy", 0.0) + 0.3 * ekman_probs.get("surprise", 0.0)
        neg_mass = (ekman_probs.get("sadness", 0.0) + ekman_probs.get("anger", 0.0) +
                    ekman_probs.get("fear", 0.0) + ekman_probs.get("disgust", 0.0) +
                    ekman_probs.get("contempt", 0.0))
        polarity = round(float(np.clip((pos_mass - neg_mass) * 1.2, -1.0, 1.0)), 3)
        subjectivity = round(float(np.clip(1.0 - ekman_probs.get("neutral", 0.0), 0.0, 1.0)), 3)

        # Token salience fallback
        words = clean_text.split()
        salience_tokens = [TokenSalience(token=w, score=round(1.0 / max(1, len(words)), 2), emotion_influence=dom_emo, sentiment_polarity=polarity) for w in words[:15]]

        # Empathy recommendation
        from src.text.nlp_emotion import TextEmotionClassifier
        advice = TextEmotionClassifier()._generate_empathy_advice(dom_emo, confidence, affect)

        top_nuances = {k: round(v, 3) for k, v in nuances.items() if v > 0.15}

        return TextEmotionResult(
            text=clean_text,
            dominant_emotion=dom_emo,
            confidence=round(confidence, 3),
            probabilities={k: round(v, 4) for k, v in ekman_probs.items()},
            nuanced_emotions=top_nuances,
            affect=affect,
            polarity=polarity,
            subjectivity=subjectivity,
            salience_tokens=salience_tokens,
            empathy_advice=advice,
            timestamp=time.time()
        )
