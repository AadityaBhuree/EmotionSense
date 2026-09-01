"""Text emotion recognition, conversational dialogue affect analysis, and transformer backends."""

from src.text.nlp_emotion import TextEmotionClassifier
from src.text.conversation_analyzer import ConversationAffectAnalyzer
from src.text.transformer_emotion import TransformerEmotionClassifier
from src.text.hybrid_classifier import HybridEmotionClassifier

__all__ = [
    "TextEmotionClassifier",
    "ConversationAffectAnalyzer",
    "TransformerEmotionClassifier",
    "HybridEmotionClassifier",
]
