"""Unit tests for TransformerEmotionClassifier and HybridEmotionClassifier."""

import pytest
from src.text.transformer_emotion import TransformerEmotionClassifier
from src.text.hybrid_classifier import HybridEmotionClassifier
from src.core.types import TextEmotionResult


def test_transformer_classifier_availability():
    clf = TransformerEmotionClassifier()
    assert isinstance(clf.is_available, bool)


def test_hybrid_classifier_lexical_mode():
    clf = HybridEmotionClassifier(mode="lexical")
    res = clf.analyze_text("I am overjoyed and so happy with this launch! 🎉")
    assert isinstance(res, TextEmotionResult)
    assert res.dominant_emotion == "joy"
    assert res.affect.valence > 0.0
    assert clf.last_latency_ms >= 0.0


def test_hybrid_classifier_mode_switch():
    clf = HybridEmotionClassifier(mode="lexical")
    assert clf.mode == "lexical"
    
    clf.set_mode("transformer")
    assert clf.mode == "transformer"

    clf.set_mode("hybrid")
    assert clf.mode == "hybrid"


def test_hybrid_classifier_hybrid_mode():
    clf = HybridEmotionClassifier(mode="hybrid")
    res = clf.analyze_text("This product is completely broken, I am furious! 😡")
    assert isinstance(res, TextEmotionResult)
    assert res.dominant_emotion == "anger"
    assert res.affect.valence < 0.0
