"""Unit and integration tests for Text Emotion Classifier & Conversational Analyzer."""

import pytest
from src.text.nlp_emotion import TextEmotionClassifier
from src.text.conversation_analyzer import ConversationAffectAnalyzer
from src.core.types import TextEmotionResult, DialogueEmotionSummary, AffectVector
from src.fusion.multimodal_fusion import MultimodalFusionEngine


@pytest.fixture
def classifier():
    return TextEmotionClassifier()


@pytest.fixture
def conv_analyzer(classifier):
    return ConversationAffectAnalyzer(classifier)


def test_joy_emotion_detection(classifier):
    text = "I am so happy and overjoyed with this fantastic news! 🎉"
    res = classifier.analyze_text(text)
    assert res.dominant_emotion == "joy"
    assert res.confidence > 0.4
    assert res.affect.valence > 0.3
    assert res.polarity > 0.0
    assert len(res.salience_tokens) > 0


def test_anger_emotion_detection(classifier):
    text = "I am furious and outraged by this terrible scam! Unacceptable! 😡"
    res = classifier.analyze_text(text)
    assert res.dominant_emotion == "anger"
    assert res.affect.valence < 0.0
    assert res.affect.arousal > 0.2


def test_sadness_emotion_detection(classifier):
    text = "I feel completely devastated, heartbroken and hopeless. 😭"
    res = classifier.analyze_text(text)
    assert res.dominant_emotion == "sadness"
    assert res.affect.valence < -0.3


def test_fear_emotion_detection(classifier):
    text = "I am terrified and panicked about this dangerous threat."
    res = classifier.analyze_text(text)
    assert res.dominant_emotion in ("fear", "anxiety")
    assert res.affect.valence < 0.0


def test_negation_inversion(classifier):
    pos_text = "I am happy with the outcome."
    neg_text = "I am not happy with the outcome."
    
    pos_res = classifier.analyze_text(pos_text)
    neg_res = classifier.analyze_text(neg_text)
    
    assert pos_res.dominant_emotion == "joy"
    assert neg_res.dominant_emotion != "joy"
    assert neg_res.affect.valence < pos_res.affect.valence


def test_emoji_affect_extraction(classifier):
    text = "Just landed! 🥳✨🚀"
    res = classifier.analyze_text(text)
    assert "🥳" in res.emojis_detected
    assert res.affect.valence > 0.2


def test_vad_bounds(classifier):
    text = "SOMETHING COMPLETELY INSANE AND OVERWHELMINGLY EXTREME!!!!!! 🤯🔥"
    res = classifier.analyze_text(text)
    assert -1.0 <= res.affect.valence <= 1.0
    assert -1.0 <= res.affect.arousal <= 1.0
    assert -1.0 <= res.affect.dominance <= 1.0


def test_conversation_transcript_parsing(conv_analyzer):
    transcript = """[10:00 AM] Alex: Hey, I am really upset about this delay!
[10:01 AM] Support: I am so sorry, let me fix it right away.
[10:02 AM] Alex: Thank you so much, you are wonderful! 🎉"""

    summary = conv_analyzer.parse_and_analyze_transcript(transcript)
    assert summary.total_turns == 3
    assert len(summary.speakers) == 2
    assert "Alex" in summary.speakers
    assert "Support" in summary.speakers
    assert len(summary.emotional_trajectory) == 3
    assert 0.0 <= summary.rapport_empathy_score <= 1.0


def test_batch_message_analysis(conv_analyzer):
    messages = [
        "I love this tool! ❤️",
        "It broke my build, absolutely terrible.",
        "Status is nominal."
    ]
    results, df = conv_analyzer.analyze_batch_messages(messages)
    assert len(results) == 3
    assert len(df) == 3
    assert "Dominant Emotion" in df.columns
    assert "Valence" in df.columns


def test_multimodal_fusion_with_text(classifier):
    engine = MultimodalFusionEngine(window_size=5)
    text_res = classifier.analyze_text("I am feeling great today! 🎉")
    
    state = engine.fuse(vision=None, voice=None, text=text_res)
    assert state.text is not None
    assert state.dominant_emotion == "joy"
    assert state.affect.valence > 0.0
