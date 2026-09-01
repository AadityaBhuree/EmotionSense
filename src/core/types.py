"""Domain types, dataclasses, and schema models for EmotionSense."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
import time


class EmotionCategory(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    CONTEMPT = "contempt"


@dataclass
class FacialActionUnits:
    """Facial Action Units (FACS) tracking intensities (0.0 - 1.0)."""
    inner_brow_raiser: float = 0.0     # AU1 (Surprise, Fear, Sadness)
    outer_brow_raiser: float = 0.0     # AU2 (Surprise)
    brow_lowerer: float = 0.0          # AU4 (Anger, Concentration)
    upper_lid_raiser: float = 0.0      # AU5 (Surprise, Fear)
    cheek_raiser: float = 0.0          # AU6 (Joy / Duchenne smile)
    lid_tightener: float = 0.0         # AU7 (Anger, Disgust)
    lip_corner_puller: float = 0.0     # AU12 (Joy / Smile)
    lip_corner_depressor: float = 0.0  # AU15 (Sadness)
    lip_pressor: float = 0.0           # AU24 (Anger, Determination)
    jaw_drop: float = 0.0              # AU26 (Surprise)
    eye_blink_left: float = 0.0        # Blink / Fatigue
    eye_blink_right: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class VisionEmotionResult:
    """Output from real-time facial mesh and micro-expression classification."""
    face_detected: bool = False
    dominant_emotion: str = "neutral"
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=lambda: {
        "joy": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "neutral": 1.0,
        "contempt": 0.0,
    })
    action_units: FacialActionUnits = field(default_factory=FacialActionUnits)
    head_pose: Dict[str, float] = field(default_factory=lambda: {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    landmarks_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["action_units"] = self.action_units.to_dict()
        return res


@dataclass
class AcousticFeatures:
    """Extracted acoustic and prosodic speech features."""
    pitch_hz: float = 0.0           # Fundamental frequency (F0)
    pitch_confidence: float = 0.0
    rms_energy: float = 0.0         # Vocal volume / intensity
    jitter_percent: float = 0.0     # Frequency micro-variability (stress/tremor)
    shimmer_percent: float = 0.0    # Amplitude micro-variability
    harmonic_to_noise: float = 0.0  # Voice quality / clarity
    tempo_bpm: float = 0.0          # Speech rate / cadence
    zero_crossing_rate: float = 0.0 # High frequency noisiness
    speech_active: bool = False

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class VoiceEmotionResult:
    """Output from acoustic tone and speech sentiment analysis."""
    dominant_emotion: str = "neutral"
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=lambda: {
        "joy": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "neutral": 1.0,
        "contempt": 0.0,
    })
    acoustics: AcousticFeatures = field(default_factory=AcousticFeatures)
    vocal_stress_level: float = 0.0  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["acoustics"] = self.acoustics.to_dict()
        return res


@dataclass
class AffectVector:
    """Continuous 3D emotional dimensional space."""
    valence: float = 0.0    # -1.0 (Negative/Unpleasant) to +1.0 (Positive/Pleasant)
    arousal: float = 0.0    # -1.0 (Calm/Deactivated) to +1.0 (Excited/Activated)
    dominance: float = 0.0  # -1.0 (Submissive/Vulnerable) to +1.0 (In Control/Empowered)

    def quadrant(self) -> str:
        if self.valence >= 0 and self.arousal >= 0:
            return "High Positive (Excited / Happy / Enthusiastic)"
        elif self.valence < 0 and self.arousal >= 0:
            return "High Negative (Angry / Stressed / Frustrated)"
        elif self.valence < 0 and self.arousal < 0:
            return "Low Negative (Depressed / Sad / Fatigued)"
        else:
            return "Low Positive (Relaxed / Content / Calm)"

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class TokenSalience:
    """Individual word or phrase salience contribution to affective prediction."""
    token: str
    score: float                      # Importance weight 0.0 to 1.0
    emotion_influence: str            # Emotion triggered (e.g. joy, anger)
    sentiment_polarity: float         # -1.0 to +1.0
    is_negated: bool = False
    is_intensified: bool = False


@dataclass
class TextEmotionResult:
    """Affective and sentiment evaluation for a single text message."""
    text: str = ""
    dominant_emotion: str = "neutral"
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=lambda: {
        "joy": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "neutral": 1.0,
        "contempt": 0.0,
    })
    nuanced_emotions: Dict[str, float] = field(default_factory=dict)
    affect: AffectVector = field(default_factory=AffectVector)
    polarity: float = 0.0             # -1.0 (Very Negative) to +1.0 (Very Positive)
    subjectivity: float = 0.0         # 0.0 (Objective) to 1.0 (Very Subjective)
    salience_tokens: List[TokenSalience] = field(default_factory=list)
    empathy_advice: str = ""
    emojis_detected: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class DialogueTurnResult:
    """Single turn / message within a multi-turn conversation."""
    turn_index: int
    speaker: str
    text: str
    timestamp_str: Optional[str] = None
    emotion_result: TextEmotionResult = field(default_factory=TextEmotionResult)
    escalation_score: float = 0.0     # 0.0 (Calm) to 1.0 (High Conflict/Frustration)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["emotion_result"] = self.emotion_result.to_dict()
        return res


@dataclass
class DialogueEmotionSummary:
    """Aggregated affective analysis across a multi-turn conversation transcript."""
    total_turns: int = 0
    speakers: List[str] = field(default_factory=list)
    turns: List[DialogueTurnResult] = field(default_factory=list)
    speaker_emotion_distribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    emotional_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    average_affect: AffectVector = field(default_factory=AffectVector)
    rapport_empathy_score: float = 0.0 # 0.0 to 1.0 (speaker emotional synchronization)
    escalation_risk: str = "Low"       # "Low", "Moderate", "High", "Critical"
    turning_points: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "speakers": self.speakers,
            "turns": [t.to_dict() for t in self.turns],
            "speaker_emotion_distribution": self.speaker_emotion_distribution,
            "emotional_trajectory": self.emotional_trajectory,
            "average_affect": self.average_affect.to_dict(),
            "rapport_empathy_score": self.rapport_empathy_score,
            "escalation_risk": self.escalation_risk,
            "turning_points": self.turning_points,
        }


@dataclass
class MultimodalEmotionState:
    """Unified temporal affect and behavioral state."""
    timestamp: float = field(default_factory=time.time)
    dominant_emotion: str = "neutral"
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=lambda: {
        "joy": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "neutral": 1.0,
        "contempt": 0.0,
    })
    affect: AffectVector = field(default_factory=AffectVector)
    engagement_index: float = 0.0   # 0.0 (Disengaged) to 1.0 (Hyper-focused)
    fatigue_level: float = 0.0      # 0.0 (Fresh) to 1.0 (Exhausted)
    attention_score: float = 0.0    # 0.0 (Distracted) to 1.0 (Direct eye contact)
    vision: Optional[VisionEmotionResult] = None
    voice: Optional[VoiceEmotionResult] = None
    text: Optional[TextEmotionResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "dominant_emotion": self.dominant_emotion,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "affect": self.affect.to_dict(),
            "engagement_index": self.engagement_index,
            "fatigue_level": self.fatigue_level,
            "attention_score": self.attention_score,
            "quadrant": self.affect.quadrant(),
            "text": self.text.to_dict() if self.text else None,
        }


@dataclass
class SessionRecord:
    """Historical session data point collection for persistence and reporting."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    samples_count: int = 0
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    average_affect: Dict[str, float] = field(default_factory=lambda: {"valence": 0.0, "arousal": 0.0, "dominance": 0.0})
    dominant_emotion_distribution: Dict[str, float] = field(default_factory=dict)
    average_engagement: float = 0.0
    average_fatigue: float = 0.0
    average_attention: float = 0.0
    key_moments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

