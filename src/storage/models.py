"""Data models and type definitions for EmotionSense database persistence."""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List


class AssessmentType(str, Enum):
    """Clinical and enterprise assessment classifications."""
    GENERAL_AFFECT = "general_affect"
    CLINICAL_SCREENING = "clinical_screening"
    TALENT_INTERVIEW = "talent_interview"
    WELLNESS_TRACKING = "wellness_tracking"
    RESEARCH_STUDY = "research_study"
    CUSTOMER_EXPERIENCE = "customer_experience"
    DYADIC_INTERVIEW = "dyadic_interview"
    GROUP_DYNAMICS = "group_dynamics"

    @classmethod
    def display_names(cls) -> Dict[str, str]:
        return {
            cls.GENERAL_AFFECT.value: "General Affect Telemetry",
            cls.CLINICAL_SCREENING.value: "Clinical Screening & Diagnosis",
            cls.TALENT_INTERVIEW.value: "Talent / Interview Assessment",
            cls.WELLNESS_TRACKING.value: "Mental Wellness Tracking",
            cls.RESEARCH_STUDY.value: "Academic / Cognitive Research",
            cls.CUSTOMER_EXPERIENCE.value: "Customer Experience Analysis",
            cls.DYADIC_INTERVIEW.value: "Dyadic / Two-Person Interview Assessment",
            cls.GROUP_DYNAMICS.value: "Group Affect & Collaborative Dynamics",
        }


@dataclass
class SessionMetadata:
    """Enterprise metadata for a recorded session."""
    session_id: str
    subject_id: Optional[str] = None
    subject_name: Optional[str] = None
    assessment_type: str = AssessmentType.GENERAL_AFFECT.value
    evaluator: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "subject_name": self.subject_name,
            "assessment_type": self.assessment_type,
            "evaluator": self.evaluator,
            "notes": self.notes,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetadata":
        tags = data.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = [t.strip() for t in tags.split(",") if t.strip()]
        return cls(
            session_id=data.get("session_id", ""),
            subject_id=data.get("subject_id"),
            subject_name=data.get("subject_name"),
            assessment_type=data.get("assessment_type", AssessmentType.GENERAL_AFFECT.value),
            evaluator=data.get("evaluator"),
            notes=data.get("notes"),
            tags=tags if isinstance(tags, list) else [],
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class StoredSession:
    """Consolidated representation of a persisted session."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    samples_count: int = 0
    duration_seconds: float = 0.0
    average_valence: float = 0.0
    average_arousal: float = 0.0
    average_dominance: float = 0.0
    average_engagement: float = 0.0
    average_fatigue: float = 0.0
    average_attention: float = 0.0
    dominant_emotion: str = "neutral"
    dominant_emotion_distribution: Dict[str, float] = field(default_factory=dict)
    key_moments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: SessionMetadata = field(default_factory=lambda: SessionMetadata(session_id=""))
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "samples_count": self.samples_count,
            "duration_seconds": self.duration_seconds,
            "average_valence": self.average_valence,
            "average_arousal": self.average_arousal,
            "average_dominance": self.average_dominance,
            "average_engagement": self.average_engagement,
            "average_fatigue": self.average_fatigue,
            "average_attention": self.average_attention,
            "dominant_emotion": self.dominant_emotion,
            "dominant_emotion_distribution": self.dominant_emotion_distribution,
            "key_moments": self.key_moments,
            "metadata": self.metadata.to_dict(),
            "anomalies": self.anomalies,
            "timeline": self.timeline,
        }
