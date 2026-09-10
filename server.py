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

from src.core.types import MultimodalEmotionState, AffectVector, SessionRecord
from src.text import HybridEmotionClassifier, ConversationAffectAnalyzer
from src.fusion.anomaly_detector import AffectiveAnomalyDetector
from src.utils.report_generator import DiagnosticReportGenerator
from src.audio.speech_transcriber import LiveSpeechTranscriber
from src.storage import SessionDatabase


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


# Request & Response Schemas
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
