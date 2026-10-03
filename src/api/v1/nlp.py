"""Affective NLP and Dialogue Trajectory Router."""

import time
from fastapi import APIRouter, HTTPException

from src.core.types import MultimodalEmotionState, AffectVector, SessionRecord
from src.fusion.anomaly_detector import AffectiveAnomalyDetector
from src.utils.report_generator import DiagnosticReportGenerator
from src.api.deps import classifier, conversation_analyzer
from src.api.v1.schemas import (
    SingleTextRequest,
    DialogueTranscriptRequest,
    BatchTextRequest,
    AnomalyDetectionRequest,
    DiagnosticReportRequest,
)

router = APIRouter(tags=["Affective NLP"])


@router.post("/api/v1/predict-text")
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


@router.post("/api/v1/analyze-dialogue")
async def analyze_dialogue_transcript(request: DialogueTranscriptRequest):
    """Parses a multi-turn conversation transcript and calculates mood trajectory & escalation risk."""
    if not request.transcript or not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    summary = conversation_analyzer.parse_and_analyze_transcript(request.transcript)
    return {
        "status": "success",
        "data": summary.to_dict(),
    }


@router.post("/api/v1/batch-predict")
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


@router.post("/api/v1/detect-anomalies")
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


@router.post("/api/v1/generate-diagnostic-report")
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
