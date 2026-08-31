"""Conversational Dialogue and Multi-Message Affective Trajectory Analyzer.

Parses multi-turn transcripts, chat threads, customer support logs, and batch text messages,
calculating speaker-level affect distribution, emotional escalation, empathy synchronization,
and conversational inflection points.
"""

import re
import time
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd

from src.core.types import (
    TextEmotionResult,
    DialogueTurnResult,
    DialogueEmotionSummary,
    AffectVector,
)
from src.text.nlp_emotion import TextEmotionClassifier


class ConversationAffectAnalyzer:
    """Enterprise multi-message and multi-turn conversational emotion engine."""

    def __init__(self, classifier: Optional[TextEmotionClassifier] = None):
        """Initializes the conversation analyzer with an NLP emotion classifier."""
        self.classifier = classifier or TextEmotionClassifier()

    def parse_and_analyze_transcript(self, raw_text: str) -> DialogueEmotionSummary:
        """Parses a multi-line conversation transcript and performs turn-by-turn affective analysis."""
        if not raw_text or not raw_text.strip():
            return DialogueEmotionSummary()

        turns = self._parse_transcript_lines(raw_text)
        if not turns:
            return DialogueEmotionSummary()

        speakers = list(dict.fromkeys([t.speaker for t in turns]))
        speaker_emotions: Dict[str, Dict[str, float]] = {s: {} for s in speakers}
        speaker_turn_counts: Dict[str, int] = {s: 0 for s in speakers}
        
        emotional_trajectory: List[Dict[str, Any]] = []
        turning_points: List[Dict[str, Any]] = []

        total_valence = 0.0
        total_arousal = 0.0
        total_dominance = 0.0

        prev_affect: Optional[AffectVector] = None
        prev_dominant: Optional[str] = None
        consecutive_negative_turns = 0

        for i, turn in enumerate(turns):
            emo_res = turn.emotion_result
            speaker = turn.speaker
            speaker_turn_counts[speaker] += 1

            # Accumulate speaker emotion frequencies
            dom_emo = emo_res.dominant_emotion
            speaker_emotions[speaker][dom_emo] = speaker_emotions[speaker].get(dom_emo, 0) + 1

            # VAD totals
            total_valence += emo_res.affect.valence
            total_arousal += emo_res.affect.arousal
            total_dominance += emo_res.affect.dominance

            # Calculate turn escalation score (anger/frustration/fear + high arousal)
            neg_intensity = emo_res.probabilities.get("anger", 0.0) * 1.3 + \
                            emo_res.probabilities.get("fear", 0.0) * 1.0 + \
                            emo_res.probabilities.get("disgust", 0.0) * 0.9 + \
                            emo_res.probabilities.get("contempt", 0.0) * 1.1
            arousal_boost = max(0.0, emo_res.affect.arousal) * 0.4
            escalation_score = min(1.0, max(0.0, neg_intensity + arousal_boost))
            turn.escalation_score = round(escalation_score, 3)

            # Track negative streaks for escalation risk
            if emo_res.affect.valence < -0.2:
                consecutive_negative_turns += 1
            else:
                consecutive_negative_turns = max(0, consecutive_negative_turns - 1)

            # Detect Turning Points (shift in valence >= 0.6 or shift from positive to negative)
            if prev_affect is not None:
                valence_delta = emo_res.affect.valence - prev_affect.valence
                if abs(valence_delta) >= 0.55 or (prev_dominant != dom_emo and emo_res.confidence >= 0.6):
                    turning_points.append({
                        "turn_index": turn.turn_index,
                        "speaker": speaker,
                        "from_emotion": prev_dominant,
                        "to_emotion": dom_emo,
                        "valence_delta": round(valence_delta, 3),
                        "text_snippet": turn.text[:60] + ("..." if len(turn.text) > 60 else ""),
                    })

            prev_affect = emo_res.affect
            prev_dominant = dom_emo

            # Add to trajectory timeline
            emotional_trajectory.append({
                "turn": turn.turn_index,
                "speaker": speaker,
                "dominant_emotion": dom_emo,
                "confidence": emo_res.confidence,
                "valence": emo_res.affect.valence,
                "arousal": emo_res.affect.arousal,
                "dominance": emo_res.affect.dominance,
                "polarity": emo_res.polarity,
                "escalation_score": turn.escalation_score,
                "text": turn.text,
            })

        # Average Affect across conversation
        n = max(1, len(turns))
        avg_affect = AffectVector(
            valence=round(total_valence / n, 3),
            arousal=round(total_arousal / n, 3),
            dominance=round(total_dominance / n, 3),
        )

        # Normalize speaker emotion distributions to percentages
        norm_speaker_emotions: Dict[str, Dict[str, float]] = {}
        for spk, emos in speaker_emotions.items():
            total_spk_turns = max(1, speaker_turn_counts[spk])
            norm_speaker_emotions[spk] = {
                k: round(v / total_spk_turns, 3) for k, v in emos.items()
            }

        # Calculate Rapport / Empathy Score (alignment of valence trajectory when multiple speakers)
        rapport_score = self._compute_rapport_score(turns, speakers)

        # Determine Escalation Risk Level
        max_escalation = max([t.escalation_score for t in turns], default=0.0)
        if max_escalation >= 0.75 or consecutive_negative_turns >= 3:
            escalation_risk = "Critical"
        elif max_escalation >= 0.5 or consecutive_negative_turns >= 2:
            escalation_risk = "High"
        elif max_escalation >= 0.3:
            escalation_risk = "Moderate"
        else:
            escalation_risk = "Low"

        return DialogueEmotionSummary(
            total_turns=len(turns),
            speakers=speakers,
            turns=turns,
            speaker_emotion_distribution=norm_speaker_emotions,
            emotional_trajectory=emotional_trajectory,
            average_affect=avg_affect,
            rapport_empathy_score=round(rapport_score, 3),
            escalation_risk=escalation_risk,
            turning_points=turning_points,
        )

    def analyze_batch_messages(self, messages: List[str]) -> Tuple[List[TextEmotionResult], pd.DataFrame]:
        """Analyzes a batch of independent text messages and returns results with summary DataFrame."""
        results: List[TextEmotionResult] = []
        records: List[Dict[str, Any]] = []

        for i, msg in enumerate(messages):
            if not msg or not msg.strip():
                continue
            res = self.classifier.analyze_text(msg)
            results.append(res)
            records.append({
                "Index": i + 1,
                "Message": msg,
                "Dominant Emotion": res.dominant_emotion.upper(),
                "Confidence (%)": round(res.confidence * 100, 1),
                "Valence": res.affect.valence,
                "Arousal": res.affect.arousal,
                "Polarity": res.polarity,
                "Subjectivity": res.subjectivity,
                "Emojis": " ".join(res.emojis_detected),
            })

        df = pd.DataFrame(records)
        return results, df

    def _parse_transcript_lines(self, text: str) -> List[DialogueTurnResult]:
        """Parses multi-line text into structured DialogueTurnResult items with speaker tagging."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        turns: List[DialogueTurnResult] = []
        turn_counter = 1

        # Patterns for standard chat lines:
        # Pattern A: [10:15 AM] Alex: Hello!
        # Pattern B: Alex: Hello!
        # Pattern C: <User>: Hello!
        # Pattern D: - Speaker: Hello!
        chat_pattern_with_time = re.compile(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\]?\s*[-–—]?\s*([^:\n]+):\s*(.*)$")
        chat_pattern_simple = re.compile(r"^<?([A-Za-z0-9_\s\.\-]{1,25})>?:?\s*[:]\s*(.*)$")

        for line in lines:
            timestamp_str = None
            speaker = f"Speaker {(turn_counter % 2) + 1}"
            message_text = line

            m_time = chat_pattern_with_time.match(line)
            if m_time:
                timestamp_str = m_time.group(1).strip()
                speaker = m_time.group(2).strip()
                message_text = m_time.group(3).strip()
            else:
                m_simp = chat_pattern_simple.match(line)
                if m_simp:
                    speaker = m_simp.group(1).strip()
                    message_text = m_simp.group(2).strip()
                else:
                    # Alternating default speakers for untagged lines
                    if len(lines) > 1:
                        speaker = f"Speaker {1 if (turn_counter % 2 == 1) else 2}"

            # Run single text affect classifier
            emo_res = self.classifier.analyze_text(message_text)

            turns.append(DialogueTurnResult(
                turn_index=turn_counter,
                speaker=speaker,
                text=message_text,
                timestamp_str=timestamp_str,
                emotion_result=emo_res,
            ))
            turn_counter += 1

        return turns

    def _compute_rapport_score(self, turns: List[DialogueTurnResult], speakers: List[str]) -> float:
        """Computes emotional synchronization and empathy alignment score between speakers."""
        if len(speakers) < 2 or len(turns) < 3:
            return 0.75  # Default baseline

        # Measure positive co-valence across alternating speaker turns
        valences = [t.emotion_result.affect.valence for t in turns]
        deltas = [abs(valences[i] - valences[i-1]) for i in range(1, len(valences))]
        avg_delta = sum(deltas) / len(deltas)

        # Lower delta indicates higher affective synchrony / mirroring
        sync_score = max(0.1, min(1.0, 1.0 - (avg_delta * 0.5)))
        return sync_score
