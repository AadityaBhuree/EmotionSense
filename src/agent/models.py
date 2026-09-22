"""Data models for Phase 9: Multimodal Agentic Reasoning & Clinical Copilot."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ClinicalRiskLevel(str, Enum):
    """Stratified affective risk levels."""
    MINIMAL = "MINIMAL"
    MILD_DISTRESS = "MILD_DISTRESS"
    MODERATE_RISK = "MODERATE_RISK"
    ACUTE_CRISIS = "ACUTE_CRISIS"
    COGNITIVE_OVERLOAD = "COGNITIVE_OVERLOAD"


class InterventionUrgency(str, Enum):
    """Urgency level for clinical or coaching interventions."""
    ROUTINE = "ROUTINE"
    RECOMMENDED = "RECOMMENDED"
    URGENT = "URGENT"
    IMMEDIATE = "IMMEDIATE"


class AffectiveObservation(BaseModel):
    """Structured clinical finding derived from multimodal signals."""
    domain: str = Field(..., description="'facial', 'acoustic', 'semantic', 'dyadic', or 'systemic'")
    finding: str = Field(..., description="Objective description of the affective phenomenon")
    clinical_significance: str = Field(..., description="Diagnostic interpretation of the finding")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


class InterventionRecommendation(BaseModel):
    """Actionable recommendation for clinician, counselor, or talent assessor."""
    action: str = Field(..., description="Recommended therapeutic or evaluative action")
    urgency: InterventionUrgency = Field(default=InterventionUrgency.ROUTINE)
    rationale: str = Field(..., description="Underlying multimodal evidence supporting this action")
    contraindications: Optional[str] = Field(default=None, description="Factors that would mitigate this action")


class ClinicalRiskScore(BaseModel):
    """Comprehensive affective risk stratification."""
    risk_level: ClinicalRiskLevel = Field(default=ClinicalRiskLevel.MINIMAL)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall risk composite 0-100")
    distress_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Negative valence * high arousal index")
    fatigue_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Cognitive exhaustion indicator")
    volatility_index: float = Field(default=0.0, ge=0.0, le=1.0, description="Affective trajectory instability")
    primary_concerns: List[str] = Field(default_factory=list, description="Top identified risk drivers")


class TurningPoint(BaseModel):
    """Key conversational or emotional inflection point."""
    timestamp_sec: float = Field(..., description="Time of inflection in session seconds")
    event: str = Field(..., description="Brief label, e.g., 'Sudden Valence Drop'")
    description: str = Field(..., description="Contextual narrative explaining the shift")
    valence_delta: float = Field(default=0.0, description="Change in valence")
    arousal_delta: float = Field(default=0.0, description="Change in arousal")


class DiagnosticSynthesis(BaseModel):
    """Comprehensive clinical evaluation produced by the reasoning agent."""
    session_id: str
    subject_id: Optional[str] = "Anonymous"
    assessment_type: str = "General Assessment"
    executive_summary: str = Field(..., description="High-level clinical summary of the session")
    risk_assessment: ClinicalRiskScore
    observations: List[AffectiveObservation] = Field(default_factory=list)
    interventions: List[InterventionRecommendation] = Field(default_factory=list)
    turning_points: List[TurningPoint] = Field(default_factory=list)
    prognosis: str = Field(default="Stable affective baseline observed.", description="Clinical outlook")
    provider_used: str = Field(default="RuleBasedExpert", description="LLM/SLM provider that synthesized result")
    model_name: str = Field(default="expert-deterministic-v1")
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentChatMessage(BaseModel):
    """A message in an interactive clinical copilot conversation."""
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    references: List[str] = Field(default_factory=list, description="Referenced anomalies or timestamps")


class ChatCopilotQuery(BaseModel):
    """Query payload sent to the in-studio clinical copilot."""
    session_id: str
    question: str
    history: List[AgentChatMessage] = Field(default_factory=list)
    include_telemetry: bool = Field(default=True)


class ChatCopilotResponse(BaseModel):
    """Response returned by the clinical copilot."""
    answer: str
    risk_notes: Optional[str] = None
    referenced_anomalies: List[str] = Field(default_factory=list)
    suggested_followups: List[str] = Field(default_factory=list)
    provider_used: str = "RuleBasedExpert"


class LLMProviderConfig(BaseModel):
    """Configuration options for pluggable LLM/SLM providers."""
    provider_name: str = Field(default="rule_based", description="'ollama', 'openai', 'gemini', 'rule_based'")
    model_name: str = Field(default="default")
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    timeout_seconds: float = Field(default=30.0, ge=1.0)
