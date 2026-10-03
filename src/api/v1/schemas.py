"""Request and Response Pydantic Schemas for EmotionSense API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SingleTextRequest(BaseModel):
    text: str = Field(..., description="Message text to analyze", json_schema_extra={"example": "I am so happy and excited for our product launch! 🎉"})
    mode: Optional[str] = Field("hybrid", description="Inference mode: lexical, transformer, or hybrid")


class DialogueTranscriptRequest(BaseModel):
    transcript: str = Field(
        ...,
        description="Multi-turn conversation transcript",
        json_schema_extra={"example": "[10:00] User: I have an urgent issue!\n[10:01] Agent: Fixing it right now!"}
    )


class BatchTextRequest(BaseModel):
    messages: List[str] = Field(..., description="List of messages to analyze")


class AnomalyDetectionRequest(BaseModel):
    frames: List[Dict[str, Any]] = Field(
        ...,
        description="List of temporal affect frames containing dominant_emotion, affect, confidence, engagement, etc.",
    )


class DiagnosticReportRequest(BaseModel):
    session_id: str = Field("session_export", description="Unique session identifier")
    frames: List[Dict[str, Any]] = Field(..., description="Timeline affect frames")
    key_moments: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Optional key moments")


class DyadicAnalysisRequest(BaseModel):
    participant_a_samples: List[Dict[str, Any]] = Field(..., description="Temporal samples for Participant A")
    participant_b_samples: List[Dict[str, Any]] = Field(..., description="Temporal samples for Participant B")
    diarization: Optional[Dict[str, Any]] = Field(None, description="Optional diarization result dictionary")


class AudioDiarizeRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio track (WAV, MP3)")
    sample_rate: Optional[int] = Field(16000, description="Target audio sample rate in Hz")
    num_speakers: Optional[int] = Field(2, description="Expected number of speakers")


class MetadataUpdateRequest(BaseModel):
    subject_id: Optional[str] = Field(None, description="Candidate or patient ID")
    subject_name: Optional[str] = Field(None, description="Candidate or patient full name")
    assessment_type: Optional[str] = Field("general_affect", description="Assessment category")
    evaluator: Optional[str] = Field(None, description="Clinician or assessor name")
    notes: Optional[str] = Field(None, description="Clinical or behavioral notes")
    tags: Optional[List[str]] = Field(None, description="Categorization tags")


class SaveSessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Unique session ID")
    start_time: float = Field(..., description="Unix epoch start timestamp")
    end_time: Optional[float] = Field(None, description="Unix epoch end timestamp")
    samples_count: Optional[int] = Field(0, description="Total sample frames")
    timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Raw or summarized timeline frames")
    average_affect: Optional[Dict[str, float]] = Field(None, description="Mean VAD coordinates")
    dominant_emotion_distribution: Optional[Dict[str, float]] = Field(None, description="Distribution of emotions")
    average_engagement: Optional[float] = Field(0.0, description="Mean engagement index")
    average_fatigue: Optional[float] = Field(0.0, description="Mean fatigue level")
    average_attention: Optional[float] = Field(0.0, description="Mean attention score")
    key_moments: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Pivot moments")
    metadata: Optional[MetadataUpdateRequest] = Field(None, description="Session metadata")
    anomalies: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Affective anomalies")


class EdgeBenchmarkRequest(BaseModel):
    model_name: Optional[str] = Field("vision_mesh", description="Model to benchmark: vision_mesh, audio_ser, text_nlp, cross_modal_cmaf")
    provider: Optional[str] = Field(None, description="Execution provider: CPUExecutionProvider, CUDAExecutionProvider, DmlExecutionProvider")
    precision: Optional[str] = Field("INT8", description="Precision: FP32, FP16, INT8, Dynamic_INT8")
    iterations: Optional[int] = Field(20, description="Iterations for latency stress testing")


class EdgeQuantizeRequest(BaseModel):
    model_name: str = Field("vision_mesh", description="Model to quantize")
    target_precision: Optional[str] = Field("INT8", description="Target precision: FP16, INT8, Dynamic_INT8")


class AgentSynthesizeRequest(BaseModel):
    session_id: Optional[str] = Field("sess_api_default", description="Session identifier")
    subject_id: Optional[str] = Field("Anonymous", description="Candidate or patient identifier")
    assessment_type: Optional[str] = Field("General Assessment", description="Assessment type")
    timeline_samples: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Timeline samples")
    anomalies: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Anomalies detected")
    provider_name: Optional[str] = Field("rule_based", description="LLM provider: rule_based, ollama, openai, gemini")


class AgentChatRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    question: str = Field(..., description="Clinician question")
    history: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Chat history")
    provider_name: Optional[str] = Field("rule_based", description="LLM provider: rule_based, ollama, openai, gemini")


class BiometricExtractRequest(BaseModel):
    rgb_samples: List[List[float]] = Field(..., description="List of [R, G, B] mean skin intensities")
    fps: Optional[float] = Field(30.0, description="Camera video frame rate")
    valence: Optional[float] = Field(0.0, description="Affective valence (-1.0 to 1.0)")
    arousal: Optional[float] = Field(0.0, description="Affective arousal (0.0 to 1.0)")
    vocal_jitter: Optional[float] = Field(0.02, description="Acoustic vocal jitter metric")


class BiometricStressRequest(BaseModel):
    bpm: float = Field(72.0, description="Heart rate in BPM")
    rmssd_ms: float = Field(42.0, description="HRV RMSSD in ms")
    baevsky_si: float = Field(85.0, description="Baevsky Stress Index")
    rpm: float = Field(15.0, description="Respiration rate in RPM")
    valence: Optional[float] = Field(0.0, description="Affective valence (-1.0 to 1.0)")
    arousal: Optional[float] = Field(0.0, description="Affective arousal (0.0 to 1.0)")
    vocal_jitter: Optional[float] = Field(0.02, description="Acoustic vocal jitter metric")


class OculometricsRequest(BaseModel):
    left_eye_points: List[List[float]] = Field(..., description="6 (x, y) coordinates for left eye")
    right_eye_points: List[List[float]] = Field(..., description="6 (x, y) coordinates for right eye")
    pir_sample: Optional[float] = Field(0.42, description="Pupil-to-Iris Ratio (PIR)")
    yaw_deg: Optional[float] = Field(0.0, description="Gaze yaw angle in degrees")
    pitch_deg: Optional[float] = Field(0.0, description="Gaze pitch angle in degrees")
    autonomic_strain: Optional[float] = Field(0.30, description="Autonomic strain index (0.0 to 1.0)")


class CognitiveWorkloadRequest(BaseModel):
    cpr: float = Field(0.25, description="Cognitive Pupillary Response index (0.0 to 1.0)")
    perclos: float = Field(0.06, description="PERCLOS eye closure fraction (0.0 to 1.0)")
    blink_suppressed: Optional[bool] = Field(False, description="Whether blink rate is suppressed (<8 bpm)")
    fixation_duration_ms: Optional[float] = Field(350.0, description="Current fixation duration in ms")
    dispersion_area: Optional[float] = Field(0.12, description="Gaze dispersion radius")
    autonomic_strain: Optional[float] = Field(0.30, description="Sympathetic autonomic strain (0.0 to 1.0)")
    speech_pause_ratio: Optional[float] = Field(0.20, description="Acoustic speech pause ratio")


class SomatosensoryKinematicsRequest(BaseModel):
    left_shoulder: List[float] = Field(..., description="[x, y] coordinates for left shoulder")
    right_shoulder: List[float] = Field(..., description="[x, y] coordinates for right shoulder")
    nose: List[float] = Field(..., description="[x, y] coordinates for nose anchor")
    left_ear: Optional[List[float]] = Field(None, description="Optional [x, y] for left ear")
    right_ear: Optional[List[float]] = Field(None, description="Optional [x, y] for right ear")
    left_hand: Optional[List[float]] = Field(None, description="Optional [x, y] for left hand/wrist")
    right_hand: Optional[List[float]] = Field(None, description="Optional [x, y] for right hand/wrist")
    autonomic_stress: Optional[float] = Field(0.25, description="Autonomic stress strain (0.0 to 1.0)")
    cognitive_workload: Optional[float] = Field(0.30, description="Cognitive workload index (0.0 to 1.0)")
    acoustic_jitter: Optional[float] = Field(0.02, description="Acoustic vocal jitter metric")
    speech_active: Optional[bool] = Field(True, description="Whether subject is actively speaking")


class PsychomotorAgitationRequest(BaseModel):
    restlessness_score: float = Field(0.20, description="Restlessness / fidgeting score (0.0 to 1.0)")
    is_fidgeting: Optional[bool] = Field(False, description="Active fidgeting flag")
    slump_index: Optional[float] = Field(0.20, description="Postural slump index (0.0 to 1.0)")
    posture_state: Optional[str] = Field("Upright", description="Current posture state")
    primary_adaptor_type: Optional[str] = Field("None", description="Active self-touch adaptor type")
    primary_adaptor_active: Optional[bool] = Field(False, description="Whether adaptor is active")
    autonomic_stress: Optional[float] = Field(0.25, description="Autonomic cardiac stress strain")
    cognitive_workload: Optional[float] = Field(0.30, description="Cognitive workload strain")
    acoustic_jitter: Optional[float] = Field(0.02, description="Acoustic vocal jitter metric")


class ForensicCredibilityRequest(BaseModel):
    action_units: Dict[str, float] = Field(default_factory=dict, description="FACS Action Unit intensities")
    macro_emotion: Optional[str] = Field("Neutral", description="Predominant macro facial emotion")
    valence: Optional[float] = Field(0.0, description="Affective valence (-1.0 to 1.0)")
    voice_stress_index: Optional[float] = Field(0.18, description="Acoustic voice stress score (0.0 to 1.0)")
    pupil_cpr: Optional[float] = Field(1.0, description="Cognitive Pupillary Response (CPR)")
    pacifying_adaptor_active: Optional[bool] = Field(False, description="Whether a pacifying self-touch adaptor is active")
    pulse_bpm: Optional[float] = Field(72.0, description="Instantaneous pulse BPM")
    response_latency_sec: Optional[float] = Field(0.45, description="Speech response latency in seconds")


class MicroLeakageAnalysisRequest(BaseModel):
    action_units: Dict[str, float] = Field(..., description="FACS Action Unit dictionary")
    macro_emotion: Optional[str] = Field("Neutral", description="Current macro emotion")
    valence: Optional[float] = Field(0.0, description="Current valence")


class HorizonForecastRequest(BaseModel):
    valence: float = Field(0.0, description="Current affective valence (-1.0 to 1.0)")
    arousal: float = Field(0.0, description="Current affective arousal (0.0 to 1.0)")
    horizon_steps: Optional[int] = Field(5, description="Number of forward Markov steps")
    step_interval_sec: Optional[float] = Field(3.0, description="Step temporal resolution in seconds")
    telemetry: Optional[Dict[str, float]] = Field(default_factory=dict, description="Continuous physiological & cognitive telemetry")


class PolicySimulationRequest(BaseModel):
    current_state: Optional[str] = Field("Baseline_Neutral", description="Current discrete affective macro-state")
    valence: Optional[float] = Field(0.0, description="Affective valence")
    arousal: Optional[float] = Field(0.0, description="Affective arousal")
    horizon_steps: Optional[int] = Field(5, description="Forward horizon steps to simulate")
    telemetry: Optional[Dict[str, float]] = Field(default_factory=dict, description="Continuous telemetry covariates")


class HealthResponse(BaseModel):
    status: str
    version: str
    active_mode: str
    timestamp: float
