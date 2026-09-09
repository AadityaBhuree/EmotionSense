"""Asynchronous WebSocket Streaming Client for EmotionSense.

Connects to /ws/stream-affect and /ws/stream-speech endpoints for low-latency
interactive typing affect decoding and live speech audio stream telemetry.
"""

import asyncio
import base64
import json
import time
from typing import Dict, Any, Optional, AsyncIterator, List


class EmotionSenseWSClient:
    """Asynchronous WebSocket client connecting to the EmotionSense streaming microservice."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, secure: bool = False):
        protocol = "wss" if secure else "ws"
        self.base_url = f"{protocol}://{host}:{port}"

    def get_url(self, endpoint: str) -> str:
        """Constructs full WebSocket URL for a given endpoint route."""
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}"

    async def stream_affect_message(self, text: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Streams a single text phrase to /ws/stream-affect and awaits affective telemetry."""
        import websockets

        url = self.get_url("/ws/stream-affect")
        t0 = time.time()
        async with websockets.connect(url) as ws:
            await ws.send(text)
            resp_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            resp = json.loads(resp_raw)
            resp["round_trip_ms"] = round((time.time() - t0) * 1000, 2)
            return resp

    async def stream_speech_text(self, text: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Streams a transcript phrase to /ws/stream-speech and returns phonetic alignment."""
        import websockets

        url = self.get_url("/ws/stream-speech")
        t0 = time.time()
        async with websockets.connect(url) as ws:
            payload = json.dumps({"text": text})
            await ws.send(payload)
            resp_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            resp = json.loads(resp_raw)
            resp["round_trip_ms"] = round((time.time() - t0) * 1000, 2)
            return resp

    async def stream_speech_audio_chunk(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """Streams raw PCM/WAV audio bytes encoded as base64 to /ws/stream-speech."""
        import websockets

        url = self.get_url("/ws/stream-speech")
        t0 = time.time()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        payload = json.dumps({
            "audio_base64": b64_audio,
            "sample_rate": sample_rate,
        })
        async with websockets.connect(url) as ws:
            await ws.send(payload)
            resp_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            resp = json.loads(resp_raw)
            resp["round_trip_ms"] = round((time.time() - t0) * 1000, 2)
            return resp
