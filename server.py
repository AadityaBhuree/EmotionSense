"""FastAPI Microservice Backend for EmotionSense.

Exposes high-performance REST and WebSocket endpoints for text emotion prediction,
multi-turn dialogue trajectory analysis, batch datasets, and real-time streaming telemetry.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import time

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    FastAPI = None

from src.core.types import TextEmotionResult, DialogueEmotionSummary
from src.text import HybridEmotionClassifier, ConversationAffectAnalyzer


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

# Global Classifiers
classifier = HybridEmotionClassifier(mode="hybrid")
conversation_analyzer = ConversationAffectAnalyzer(classifier)


# Request & Response Schemas
class SingleTextRequest(BaseModel):
    text: str = Field(..., description="Message text to analyze", example="I am so happy and excited for our product launch! 🎉")
    mode: Optional[str] = Field("hybrid", description="Inference mode: lexical, transformer, or hybrid")


class DialogueTranscriptRequest(BaseModel):
    transcript: str = Field(
        ...,
        description="Multi-turn conversation transcript",
        example="[10:00] User: I have an urgent issue!\n[10:01] Agent: Fixing it right now!"
    )


class BatchTextRequest(BaseModel):
    messages: List[str] = Field(..., description="List of messages to analyze")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
