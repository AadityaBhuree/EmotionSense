"""FastAPI Microservice Backend for EmotionSense.

Exposes high-performance REST and WebSocket endpoints for text emotion prediction,
multi-turn dialogue trajectory analysis, batch datasets, and real-time streaming telemetry.
Refactored into modular APIRouters under src/api/v1/.
"""

import base64
import json
import os
import time
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
except ImportError:
    FastAPI = None

from src.api.deps import (
    biometric_engine,
    chat_copilot,
    classifier,
    clinical_agent,
    conversation_analyzer,
    credibility_engine,
    db,
    edge_bench_suite,
    edge_engine,
    edge_quantizer,
    forecasting_engine,
    oculomotor_engine,
    somatosensory_engine,
    speech_transcriber,
)
from src.api.v1 import api_v1_router
from src.api.v1.schemas import HealthResponse

# Initialize FastAPI App
app = FastAPI(
    title="EmotionSense Affective Intelligence API",
    description="Enterprise Multimodal Emotion Recognition, Conversational Trajectory & 3D VAD Affect Engine",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# API Key Authentication Configuration
API_KEY_HEADER = "X-API-Key"
DEFAULT_DEV_API_KEY = "emotionsense-secure-key-2026"
EXEMPT_ROUTES = {"/health", "/docs", "/redoc", "/openapi.json"}


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Enforces API key header validation for all protected REST endpoints."""

    async def dispatch(self, request: Request, call_next):
        # Allow CORS preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Allow public documentation and health check endpoints
        if request.url.path in EXEMPT_ROUTES:
            return await call_next(request)

        # Skip WebSockets in HTTP middleware
        if request.scope.get("type") == "websocket":
            return await call_next(request)

        # Extract and validate API Key
        configured_key = os.getenv("API_KEY", DEFAULT_DEV_API_KEY)
        client_key = request.headers.get(API_KEY_HEADER)

        if not client_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                client_key = auth_header[7:].strip()

        if not client_key or client_key != configured_key:
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "detail": "Unauthorized: A valid API key must be provided via the 'X-API-Key' or 'Authorization: Bearer <key>' header.",
                },
            )

        return await call_next(request)


# 1. Register API Key Auth Middleware (inner layer)
app.add_middleware(APIKeyAuthMiddleware)

# 2. Register CORS Middleware (outer layer - executes first on incoming requests)
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
_env_origins = os.getenv("ALLOWED_ORIGINS")
allowed_origins = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else DEFAULT_ALLOWED_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 3. Mount Modular API Routers
app.include_router(api_v1_router)


# 4. System Endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint returning engine status."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        active_mode=classifier.mode,
        timestamp=time.time(),
    )


# 5. Real-Time Streaming WebSockets
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
