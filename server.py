"""FastAPI Microservice Backend for EmotionSense.

Exposes high-performance REST and WebSocket endpoints for text emotion prediction,
multi-turn dialogue trajectory analysis, batch datasets, and real-time streaming telemetry.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import time
import json
import base64

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    FastAPI = None

from src.core.types import MultimodalEmotionState, AffectVector, SessionRecord, DiarizationResult, SpeakerTurn
from src.text import HybridEmotionClassifier, ConversationAffectAnalyzer
from src.fusion.anomaly_detector import AffectiveAnomalyDetector
from src.fusion.interaction_dynamics import DyadicInteractionAnalyzer
from src.utils.report_generator import DiagnosticReportGenerator
from src.audio.speech_transcriber import LiveSpeechTranscriber
from src.audio.diarizer import AcousticDiarizer
from src.storage import SessionDatabase
from src.analytics import (
    LongitudinalProfileAnalyzer,
    generate_synthetic_cohort_benchmarks,
    BiometricEngine,
    OculomotorEngine,
)
from src.edge.runtime import ONNXEdgeInferenceEngine
from src.edge.quantizer import ModelQuantizationOptimizer
from src.edge.benchmark import EdgeBenchmarkSuite
from src.agent import ClinicalReasoningAgent, ClinicalChatCopilot, ChatCopilotQuery, LLMProviderConfig
from src.core.biometric_models import (
    StressClassification,
    PulseMeasurement,
    HRVMetrics,
    RespirationMetrics,
    BiometricTelemetry,
)
from src.core.cognitive_models import (
    WorkloadTier,
    PupillometryMetrics,
    BlinkDynamics,
    GazeTelemetry,
)


# Initialize FastAPI App
app = FastAPI(
    title="EmotionSense Affective Intelligence API",
    description="Enterprise Multimodal Emotion Recognition, Conversational Trajectory & 3D VAD Affect Engine",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Classifiers, Engines & Database
classifier = HybridEmotionClassifier(mode="hybrid")
conversation_analyzer = ConversationAffectAnalyzer(classifier)
speech_transcriber = LiveSpeechTranscriber(classifier.lexical_clf)
db = SessionDatabase()
edge_engine = ONNXEdgeInferenceEngine()
edge_quantizer = ModelQuantizationOptimizer()
edge_bench_suite = EdgeBenchmarkSuite(engine=edge_engine, quantizer=edge_quantizer)
clinical_agent = ClinicalReasoningAgent()
chat_copilot = ClinicalChatCopilot()
biometric_engine = BiometricEngine(fps=30.0)
oculomotor_engine = OculomotorEngine()


# Request & Response Schemas
class SingleTextRequest(BaseModel):
    text: str = Field(..., description="Message text to analyze", json_schema_extra={"example": "I am so happy and excited for our product launch! 🎉"})
    mode: Optional[str] = Field("hybrid", description="Inference mode: lexical, transformer, or hybrid")


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
        json_schema_extra={
            "example": [
                {
                    "timestamp": 100.0,
                    "dominant_emotion": "joy",
                    "confidence": 0.85,
                    "affect": {"valence": 0.6, "arousal": 0.5, "dominance": 0.5},
                    "engagement_index": 0.8,
                    "fatigue_level": 0.2,
                    "attention_score": 0.85
                },
                {
                    "timestamp": 102.0,
                    "dominant_emotion": "anger",
                    "confidence": 0.92,
                    "affect": {"valence": -0.75, "arousal": 0.8, "dominance": 0.7},
                    "engagement_index": 0.4,
                    "fatigue_level": 0.7,
                    "attention_score": 0.5
                }
            ]
        }
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


class HealthResponse(BaseModel):
    status: str
    version: str
    active_mode: str
    timestamp: float


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint returning engine status."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        active_mode=classifier.mode,
        timestamp=time.time(),
    )


@app.post("/api/v1/predict-text", tags=["Affective NLP"])
async def predict_single_text(request: SingleTextRequest):
    """Predicts emotions, 3D VAD coordinates, and token salience for a single text message."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    if request.mode in ("lexical", "transformer", "hybrid"):
        classifier.set_mode(request.mode)

    result = classifier.analyze_text(request.text)
    return {
        "status": "success",
        "data": result.to_dict(),
        "latency_ms": classifier.last_latency_ms,
    }


@app.post("/api/v1/analyze-dialogue", tags=["Dialogue Trajectory"])
async def analyze_dialogue_transcript(request: DialogueTranscriptRequest):
    """Parses a multi-turn conversation transcript and calculates mood trajectory & escalation risk."""
    if not request.transcript or not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    summary = conversation_analyzer.parse_and_analyze_transcript(request.transcript)
    return {
        "status": "success",
        "data": summary.to_dict(),
    }


@app.post("/api/v1/batch-predict", tags=["Batch Analytics"])
async def batch_predict(request: BatchTextRequest):
    """Analyzes a list of text messages and returns individual and aggregated statistics."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    results, df = conversation_analyzer.analyze_batch_messages(request.messages)
    distribution = df["Dominant Emotion"].value_counts().to_dict() if not df.empty else {}

    return {
        "status": "success",
        "total_messages": len(results),
        "distribution": distribution,
        "results": [r.to_dict() for r in results],
    }


@app.post("/api/v1/detect-anomalies", tags=["Affective Anomaly Sentinel"])
async def detect_anomalies(request: AnomalyDetectionRequest):
    """Evaluates a stream or session of affect states for emotional distress, valence crashes, and fatigue."""
    if not request.frames:
        raise HTTPException(status_code=400, detail="Frames list cannot be empty.")

    detector = AffectiveAnomalyDetector()
    for frame in request.frames:
        aff = frame.get("affect", {})
        st_obj = MultimodalEmotionState(
            timestamp=frame.get("timestamp", time.time()),
            dominant_emotion=frame.get("dominant_emotion", "neutral"),
            confidence=frame.get("confidence", 0.5),
            affect=AffectVector(
                valence=aff.get("valence", 0.0),
                arousal=aff.get("arousal", 0.0),
                dominance=aff.get("dominance", 0.0)
            ),
            engagement_index=frame.get("engagement_index", 0.5),
            fatigue_level=frame.get("fatigue_level", 0.2),
            attention_score=frame.get("attention_score", 0.7),
        )
        detector.process_state(st_obj)

    summary = detector.get_anomaly_summary()
    return {
        "status": "success",
        "data": summary
    }


@app.post("/api/v1/generate-diagnostic-report", tags=["Clinical Reporting"])
async def generate_diagnostic_report(request: DiagnosticReportRequest):
    """Generates standalone clinical diagnostic reports in HTML and Markdown formats."""
    if not request.frames:
        raise HTTPException(status_code=400, detail="Frames list cannot be empty.")

    detector = AffectiveAnomalyDetector()
    valences, arousals, engagements, fatigues, attentions = [], [], [], [], []
    emo_counts = {}

    for frame in request.frames:
        aff = frame.get("affect", {})
        v = aff.get("valence", 0.0)
        a = aff.get("arousal", 0.0)
        dom = aff.get("dominance", 0.0)
        emo = frame.get("dominant_emotion", "neutral")

        valences.append(v)
        arousals.append(a)
        engagements.append(frame.get("engagement_index", 0.5))
        fatigues.append(frame.get("fatigue_level", 0.2))
        attentions.append(frame.get("attention_score", 0.7))
        emo_counts[emo] = emo_counts.get(emo, 0) + 1

        st_obj = MultimodalEmotionState(
            timestamp=frame.get("timestamp", 0.0),
            dominant_emotion=emo,
            confidence=frame.get("confidence", 0.5),
            affect=AffectVector(valence=v, arousal=a, dominance=dom),
            engagement_index=frame.get("engagement_index", 0.5),
            fatigue_level=frame.get("fatigue_level", 0.2),
            attention_score=frame.get("attention_score", 0.7),
        )
        detector.process_state(st_obj)

    total = len(request.frames)
    dist = {k: v / max(1, total) for k, v in emo_counts.items()}

    session_rec = SessionRecord(
        session_id=request.session_id,
        start_time=request.frames[0].get("timestamp", 0.0),
        end_time=request.frames[-1].get("timestamp", 0.0),
        samples_count=total,
        timeline=request.frames,
        average_affect={
            "valence": sum(valences) / max(1, total),
            "arousal": sum(arousals) / max(1, total),
            "dominance": 0.0
        },
        dominant_emotion_distribution=dist,
        average_engagement=sum(engagements) / max(1, total),
        average_fatigue=sum(fatigues) / max(1, total),
        average_attention=sum(attentions) / max(1, total),
        key_moments=request.key_moments or [],
    )

    detected_events = detector.get_anomaly_summary().get("events", [])
    html_report = DiagnosticReportGenerator.generate_html_report(session_rec, detected_events)
    md_report = DiagnosticReportGenerator.generate_markdown_report(session_rec, detected_events)

    return {
        "status": "success",
        "session_id": request.session_id,
        "total_anomalies": len(detected_events),
        "html_report": html_report,
        "markdown_report": md_report,
    }


# ============================================================================
# Phase 5: Multi-Speaker Diarization & Dyadic Interaction Endpoints
# ============================================================================

@app.post("/api/analyze/dyadic", tags=["Dyadic Interaction"])
async def analyze_dyadic_interaction(request: DyadicAnalysisRequest):
    """Computes interpersonal synchrony, conversational balance, and Dyadic Rapport Score."""
    try:
        diar_res = None
        if request.diarization:
            turns = [
                SpeakerTurn(
                    speaker_id=t.get("speaker_id", "Speaker_0"),
                    start_time=float(t.get("start_time", 0.0)),
                    end_time=float(t.get("end_time", 0.0)),
                    duration=float(t.get("duration", 0.0)),
                )
                for t in request.diarization.get("turns", [])
            ]
            diar_res = DiarizationResult(
                turns=turns,
                speakers=request.diarization.get("speakers", []),
                speaker_durations=request.diarization.get("speaker_durations", {}),
                dominance_ratios=request.diarization.get("dominance_ratios", {}),
                interruption_count=int(request.diarization.get("interruption_count", 0)),
                total_speech_duration=float(request.diarization.get("total_speech_duration", 0.0)),
                total_audio_duration=float(request.diarization.get("total_audio_duration", 0.0)),
            )
        analyzer = DyadicInteractionAnalyzer()
        metrics = analyzer.analyze(request.participant_a_samples, request.participant_b_samples, diar_res)
        return {
            "status": "success",
            "metrics": metrics.to_dict(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dyadic analysis failed: {str(exc)}")


@app.post("/api/audio/diarize", tags=["Acoustic Diarization"])
async def diarize_audio(request: AudioDiarizeRequest):
    """Performs acoustic speaker diarization on base64 encoded audio."""
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        diarizer = AcousticDiarizer(sample_rate=request.sample_rate or 16000, num_speakers=request.num_speakers or 2)
        res = diarizer.diarize_file(audio_bytes)
        return {
            "status": "success",
            "diarization": res.to_dict(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speaker diarization failed: {str(exc)}")


# ============================================================================
# Session Persistence & Intelligence Endpoints
# ============================================================================

@app.get("/api/sessions", tags=["Session Persistence"])
def list_persisted_sessions(
    assessment_type: Optional[str] = Query(None, description="Filter by assessment type"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    search: Optional[str] = Query(None, description="Search term across session ID, subject name, or notes"),
    limit: int = Query(50, ge=1, le=200, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """Lists saved multimodal sessions with filtering, pagination, and metadata."""
    sessions = db.list_sessions(
        assessment_type=assessment_type,
        tag=tag,
        search_query=search,
        limit=limit,
        offset=offset
    )
    return {
        "status": "success",
        "count": len(sessions),
        "limit": limit,
        "offset": offset,
        "sessions": sessions,
    }


@app.get("/api/sessions/{session_id}", tags=["Session Persistence"])
def get_persisted_session(session_id: str, include_samples: bool = Query(True, description="Whether to include granular timeline samples")):
    """Retrieves full persisted session, aggregates, metadata, and anomaly alerts."""
    session = db.get_session(session_id, include_samples=include_samples)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session.to_dict()


@app.post("/api/sessions", tags=["Session Persistence"])
def save_persisted_session(request: SaveSessionRequest):
    """Creates or updates a session in SQLite database."""
    sess_dict = request.model_dump()
    meta_dict = sess_dict.pop("metadata", None)
    anom_list = sess_dict.pop("anomalies", None)
    saved_id = db.save_session(sess_dict, metadata=meta_dict, anomalies=anom_list)
    return {"status": "success", "session_id": saved_id}


@app.patch("/api/sessions/{session_id}/metadata", tags=["Session Persistence"])
def update_session_metadata(session_id: str, request: MetadataUpdateRequest):
    """Updates candidate/patient metadata, notes, and tags for a session."""
    existing = db.get_session(session_id, include_samples=False)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    meta_dict = request.model_dump(exclude_unset=True)
    meta_dict["session_id"] = session_id
    success = db.update_metadata(session_id, meta_dict)
    return {"status": "success" if success else "failed", "session_id": session_id}


@app.delete("/api/sessions/{session_id}", tags=["Session Persistence"])
def delete_persisted_session(session_id: str):
    """Deletes a session and cascading samples and anomalies from database."""
    deleted = db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "success", "deleted_session_id": session_id}


@app.post("/api/sessions/migrate", tags=["Session Persistence"])
def trigger_json_migration():
    """Migrates all legacy flat JSON sessions into SQLite database."""
    count = db.migrate_from_json()
    return {"status": "success", "migrated_sessions_count": count}


@app.get("/api/stats", tags=["System"])
def get_platform_statistics():
    """Returns platform-wide metrics: total sessions, samples, anomalies, and assessment types."""
    return {"status": "success", "stats": db.get_stats()}


# ============================================================================
# Phase 7: Longitudinal Trajectory & Cohort Analytics Endpoints
# ============================================================================

@app.get("/api/longitudinal/subjects", tags=["Longitudinal Analytics"])
async def list_longitudinal_subjects():
    """Returns list of distinct evaluated subjects, session counts, and date ranges."""
    try:
        subjects = db.list_distinct_subjects()
        return {
            "status": "success",
            "count": len(subjects),
            "subjects": subjects,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list subjects: {str(exc)}")


@app.get("/api/longitudinal/{subject_id}", tags=["Longitudinal Analytics"])
async def get_subject_trajectory(subject_id: str, cohort: Optional[str] = None):
    """Retrieves full longitudinal session trajectory and drift metrics for a subject."""
    try:
        points = db.get_subject_longitudinal_points(subject_id)
        benchmarks = generate_synthetic_cohort_benchmarks()
        cohort_obj = benchmarks.get(cohort) if cohort else None

        analyzer = LongitudinalProfileAnalyzer()
        profile = analyzer.analyze_profile(
            subject_id=subject_id,
            subject_name=subject_id,
            history_points=points,
            cohort=cohort_obj,
        )
        return {
            "status": "success",
            "profile": profile.to_dict(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Longitudinal analysis failed: {str(exc)}")


@app.get("/api/cohort/benchmarks", tags=["Longitudinal Analytics"])
async def get_cohort_benchmarks():
    """Returns population normative cohort benchmarks for comparative evaluation."""
    try:
        benchmarks = generate_synthetic_cohort_benchmarks()
        return {
            "status": "success",
            "benchmarks": {k: b.to_dict() for k, b in benchmarks.items()},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch benchmarks: {str(exc)}")


# =====================================================================
# Phase 8: Edge AI Acceleration & Quantization Endpoints
# =====================================================================

@app.get("/api/edge/hardware", tags=["Edge Acceleration"])
async def get_edge_hardware():
    """Returns detected hardware specifications, CPU threads, RAM, and active ONNX execution providers."""
    profile = edge_engine.get_device_profile()
    return {"status": "success", "profile": profile.to_dict()}


@app.post("/api/edge/benchmark", tags=["Edge Acceleration"])
async def benchmark_edge_model(req: EdgeBenchmarkRequest):
    """Executes latency stress testing across iterations, calculating P50/P95/P99 percentiles and FPS throughput."""
    res = edge_bench_suite.run_benchmark(
        model_name=req.model_name or "vision_mesh",
        provider=req.provider,
        precision=req.precision or "INT8",
        iterations=req.iterations or 20,
    )
    return {"status": "success", "benchmark": res.to_dict()}


@app.post("/api/edge/quantize", tags=["Edge Acceleration"])
async def quantize_edge_model(req: EdgeQuantizeRequest):
    """Computes dynamic post-training quantization metrics, size reduction ratios, and accuracy retention."""
    summary = edge_quantizer.optimize_model(
        model_name=req.model_name,
        target_precision=req.target_precision or "INT8",
    )
    return {"status": "success", "quantization": summary.to_dict()}


@app.get("/api/agent/providers", tags=["Agentic Reasoning"])
async def get_agent_providers():
    """Returns supported LLM/SLM reasoning providers and active default."""
    return {
        "status": "success",
        "providers": ["rule_based", "ollama", "openai", "gemini"],
        "active_default": "rule_based",
    }


@app.post("/api/agent/synthesize", tags=["Agentic Reasoning"])
async def synthesize_session_diagnostics(req: AgentSynthesizeRequest):
    """Executes Chain-of-Thought clinical diagnostic synthesis across multimodal session telemetry."""
    session_data = {
        "session_id": req.session_id,
        "candidate_id": req.subject_id,
        "assessment_type": req.assessment_type,
        "timeline_samples": req.timeline_samples or [],
        "anomalies": req.anomalies or [],
    }
    config = LLMProviderConfig(provider_name=req.provider_name or "rule_based")
    agent = ClinicalReasoningAgent(provider_config=config)
    synthesis = agent.synthesize_session(session_data)
    return {"status": "success", "synthesis": synthesis.model_dump()}


@app.post("/api/agent/chat", tags=["Agentic Reasoning"])
async def chat_clinical_copilot(req: AgentChatRequest):
    """Answers clinician queries grounded in multimodal session telemetry."""
    config = LLMProviderConfig(provider_name=req.provider_name or "rule_based")
    copilot = ClinicalChatCopilot(provider_config=config)
    query = ChatCopilotQuery(
        session_id=req.session_id,
        question=req.question,
    )
    ctx = {"session_id": req.session_id}
    stored = db.get_session(req.session_id)
    if stored:
        ctx["timeline_samples"] = stored.get("timeline_samples", [])
        ctx["anomalies"] = stored.get("anomalies", [])
        ctx["candidate_id"] = stored.get("candidate_id", "Anonymous")
        ctx["assessment_type"] = stored.get("assessment_type", "Assessment")

    resp = copilot.query(query, session_context=ctx)
    return {"status": "success", "response": resp.model_dump()}


@app.post("/api/biometrics/rppg", tags=["Remote Biometrics"])
async def extract_rppg_telemetry(req: BiometricExtractRequest):
    """Extract contact-free optical pulse, HRV, respiration, and autonomic stress from RGB stream."""
    engine = BiometricEngine(fps=req.fps or 30.0)
    for sample in req.rgb_samples:
        if len(sample) >= 3:
            engine.add_rgb_sample(sample[0], sample[1], sample[2])

    bvp = engine.extract_pos_bvp()
    pulse = engine.compute_pulse_from_bvp(bvp)
    rr_intervals = engine.extract_rr_intervals(bvp)
    hrv = engine.compute_hrv_metrics(rr_intervals)
    respiration = engine.estimate_respiration_rate(bvp, rr_intervals)
    stress = engine.compute_autonomic_stress(
        pulse, hrv, respiration, req.valence or 0.0, req.arousal or 0.0, req.vocal_jitter or 0.02
    )
    bvp_history = [float(x) for x in bvp[-120:]] if len(bvp) > 0 else []

    telemetry = BiometricTelemetry(
        timestamp=time.time(),
        pulse=pulse,
        hrv=hrv,
        respiration=respiration,
        autonomic_stress=stress,
        bvp_history=bvp_history,
        rr_intervals_ms=rr_intervals,
        roi_detected=True,
    )
    return {"status": "success", "telemetry": telemetry.to_dict()}


@app.post("/api/biometrics/stress", tags=["Remote Biometrics"])
async def compute_biometric_stress(req: BiometricStressRequest):
    """Compute autonomic stress classification and sympathetic/parasympathetic tone."""
    pulse = PulseMeasurement(bpm=req.bpm)
    hrv = HRVMetrics(rmssd_ms=req.rmssd_ms, baevsky_stress_index=req.baevsky_si)
    resp = RespirationMetrics(rpm=req.rpm)
    stress = BiometricEngine.compute_autonomic_stress(
        pulse, hrv, resp, req.valence or 0.0, req.arousal or 0.0, req.vocal_jitter or 0.02
    )
    return {"status": "success", "stress": stress.to_dict()}


@app.get("/api/biometrics/status", tags=["Remote Biometrics"])
async def get_biometric_status():
    """Retrieve biometric engine status, sampling configuration, and supported algorithms."""
    return {
        "status": "online",
        "algorithms": ["Plane-Orthogonal-to-Skin (POS)", "Chrominance (CHROM)", "Respiratory Sinus Arrhythmia (RSA)"],
        "default_fps": 30.0,
        "hrv_metrics": ["SDNN", "RMSSD", "pNN50", "Baevsky_Stress_Index", "HRV_Vitality_Score"],
        "autonomic_classifications": [c.value for c in StressClassification],
    }


@app.post("/api/cognitive/oculometrics", tags=["Cognitive Workload"])
async def evaluate_oculometrics(req: OculometricsRequest):
    """Evaluate eye landmarks, pupil dilation ratio, blink rate, and gaze dynamics."""
    import numpy as np
    l_pts = np.array(req.left_eye_points, dtype=np.float64)
    r_pts = np.array(req.right_eye_points, dtype=np.float64)
    snapshot = oculomotor_engine.process_frame(
        left_eye_pts=l_pts,
        right_eye_pts=r_pts,
        pir_sample=req.pir_sample or 0.42,
        yaw_deg=req.yaw_deg or 0.0,
        pitch_deg=req.pitch_deg or 0.0,
        autonomic_strain=req.autonomic_strain or 0.30,
    )
    return {"status": "success", "snapshot": snapshot.to_dict()}


@app.post("/api/cognitive/workload", tags=["Cognitive Workload"])
async def compute_cognitive_workload_endpoint(req: CognitiveWorkloadRequest):
    """Compute unified Cognitive Workload Index and NASA-TLX dimensional estimates."""
    pupil = PupillometryMetrics(cognitive_pupillary_response=req.cpr)
    blinks = BlinkDynamics(perclos=req.perclos, blink_suppressed=req.blink_suppressed or False)
    gaze = GazeTelemetry(
        fixation_duration_ms=req.fixation_duration_ms or 350.0,
        dispersion_area=req.dispersion_area or 0.12,
    )
    workload = OculomotorEngine.compute_cognitive_workload(
        pupil,
        blinks,
        gaze,
        autonomic_strain=req.autonomic_strain or 0.30,
        speech_pause_ratio=req.speech_pause_ratio or 0.20,
    )
    return {"status": "success", "workload": workload.to_dict()}


@app.get("/api/cognitive/config", tags=["Cognitive Workload"])
async def get_cognitive_config():
    """Retrieve cognitive engine calibration thresholds, EAR baseline, and NASA-TLX specifications."""
    return {
        "status": "online",
        "ear_closed_threshold": oculomotor_engine.ear_closed_threshold,
        "baseline_pir": oculomotor_engine.baseline_pir,
        "saccade_velocity_threshold_deg_s": oculomotor_engine.saccade_velocity_threshold,
        "workload_tiers": [t.value for t in WorkloadTier],
        "nasa_tlx_dimensions": ["mental_demand", "temporal_demand", "effort", "frustration"],
    }


@app.websocket("/ws/stream-affect")

async def websocket_affect_stream(websocket: WebSocket):
    """Real-time bidirectional WebSocket stream for interactive typing affect decoding."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                result = classifier.analyze_text(data)
                await websocket.send_json({
                    "text": data,
                    "dominant_emotion": result.dominant_emotion,
                    "confidence": result.confidence,
                    "valence": result.affect.valence,
                    "arousal": result.affect.arousal,
                    "dominance": result.affect.dominance,
                    "polarity": result.polarity,
                    "empathy_advice": result.empathy_advice,
                    "timestamp": time.time(),
                })
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/stream-speech")
async def websocket_speech_stream(websocket: WebSocket):
    """Bidirectional WebSocket stream for live speech transcription and phonetic affect alignment."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                res = None
                trimmed = data.strip()
                if trimmed.startswith("{") and trimmed.endswith("}"):
                    try:
                        payload = json.loads(trimmed)
                        if "audio_base64" in payload or "audio_chunk" in payload:
                            b64_str = payload.get("audio_base64") or payload.get("audio_chunk")
                            audio_bytes = base64.b64decode(b64_str)
                            sr = int(payload.get("sample_rate", 16000))
                            res = speech_transcriber.transcribe_audio_bytes(audio_bytes, sample_rate=sr)
                        elif "text" in payload:
                            res = speech_transcriber.transcribe_text_stream(str(payload["text"]))
                    except Exception:
                        pass

                if res is None:
                    res = speech_transcriber.transcribe_text_stream(data)

                await websocket.send_json({
                    "transcript": res.full_transcript,
                    "dominant_emotion": res.text_emotion.dominant_emotion if res.text_emotion else "neutral",
                    "confidence": res.text_emotion.confidence if res.text_emotion else 0.0,
                    "valence": res.text_emotion.affect.valence if res.text_emotion else 0.0,
                    "arousal": res.text_emotion.affect.arousal if res.text_emotion else 0.0,
                    "tokens": [
                        {
                            "word": t.word,
                            "emotion_cue": t.emotion_cue,
                            "confidence": t.confidence,
                            "pitch_hz": t.pitch_hz,
                        }
                        for t in res.tokens
                    ],
                    "timestamp": res.timestamp,
                })
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
