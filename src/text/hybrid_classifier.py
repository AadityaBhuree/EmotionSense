"""Hybrid Ensemble Emotion Classifier.

Combines rule-based lexical affect processing (low latency, high emoji/negation precision)
with deep neural transformer models (contextual RoBERTa embeddings) into a unified classifier.
"""

import time
from typing import Dict, Optional, Any
import numpy as np

from src.core.types import TextEmotionResult, AffectVector
from src.text.nlp_emotion import TextEmotionClassifier
from src.text.transformer_emotion import TransformerEmotionClassifier


class HybridEmotionClassifier:
    """Enterprise hybrid classifier supporting Lexical, Transformer, and Ensembled inference."""

    def __init__(self, mode: str = "hybrid", model_name: str = "SamLowe/roberta-base-go_emotions"):
        """Initializes the hybrid emotion engine."""
        self.mode = mode  # "lexical", "transformer", or "hybrid"
        self.lexical_clf = TextEmotionClassifier()
        self.transformer_clf = TransformerEmotionClassifier(model_name=model_name)
        self.last_latency_ms: float = 0.0

    def set_mode(self, mode: str) -> None:
        """Sets the active inference mode."""
        if mode in ("lexical", "transformer", "hybrid"):
            self.mode = mode

    def analyze_text(self, text: str) -> TextEmotionResult:
        """Analyzes text using the configured mode."""
        t0 = time.time()

        if self.mode == "lexical":
            res = self.lexical_clf.analyze_text(text)
        elif self.mode == "transformer":
            res = self.transformer_clf.predict(text)
        else:
            # Hybrid Ensemble Mode
            res_lex = self.lexical_clf.analyze_text(text)
            
            # If transformer available and loaded, blend predictions
            if self.transformer_clf.is_available:
                try:
                    res_trans = self.transformer_clf.predict(text)
                    
                    # 55% Transformer + 45% Lexical
                    w_trans = 0.55
                    w_lex = 0.45

                    blended_probs: Dict[str, float] = {}
                    for emo in res_lex.probabilities:
                        p_t = res_trans.probabilities.get(emo, 0.0)
                        p_l = res_lex.probabilities.get(emo, 0.0)
                        blended_probs[emo] = float(w_trans * p_t + w_lex * p_l)

                    # Normalize
                    tot = sum(blended_probs.values())
                    if tot > 0:
                        blended_probs = {k: float(v / tot) for k, v in blended_probs.items()}

                    dom_emo = max(blended_probs, key=blended_probs.get)
                    confidence = blended_probs[dom_emo]

                    # Blend affect VAD
                    v = float(np.clip(w_trans * res_trans.affect.valence + w_lex * res_lex.affect.valence, -1.0, 1.0))
                    a = float(np.clip(w_trans * res_trans.affect.arousal + w_lex * res_lex.affect.arousal, -1.0, 1.0))
                    d = float(np.clip(w_trans * res_trans.affect.dominance + w_lex * res_lex.affect.dominance, -1.0, 1.0))

                    affect = AffectVector(valence=round(v, 3), arousal=round(a, 3), dominance=round(d, 3))
                    polarity = round(float(np.clip(w_trans * res_trans.polarity + w_lex * res_lex.polarity, -1.0, 1.0)), 3)

                    res = TextEmotionResult(
                        text=text,
                        dominant_emotion=dom_emo,
                        confidence=round(confidence, 3),
                        probabilities={k: round(v, 4) for k, v in blended_probs.items()},
                        nuanced_emotions=res_lex.nuanced_emotions,
                        affect=affect,
                        polarity=polarity,
                        subjectivity=res_lex.subjectivity,
                        salience_tokens=res_lex.salience_tokens,
                        empathy_advice=res_lex.empathy_advice,
                        emojis_detected=res_lex.emojis_detected,
                        timestamp=time.time(),
                    )
                except Exception:
                    res = res_lex
            else:
                res = res_lex

        self.last_latency_ms = round((time.time() - t0) * 1000, 2)
        return res
