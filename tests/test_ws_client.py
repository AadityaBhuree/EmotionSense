"""Unit tests for WebSocket streaming client utility."""

import pytest
import numpy as np
import base64
import json
from src.utils.ws_client import EmotionSenseWSClient
from server import app

try:
    from starlette.testclient import TestClient
except ImportError:
    from fastapi.testclient import TestClient


def test_ws_client_url_construction():
    client = EmotionSenseWSClient(host="localhost", port=8000, secure=False)
    assert client.get_url("/ws/stream-affect") == "ws://localhost:8000/ws/stream-affect"
    assert client.get_url("ws/stream-speech") == "ws://localhost:8000/ws/stream-speech"

    client_ssl = EmotionSenseWSClient(host="emotionsense.ai", port=443, secure=True)
    assert client_ssl.get_url("/ws/stream-affect") == "wss://emotionsense.ai:443/ws/stream-affect"


def test_ws_client_stream_affect_flow():
    client = TestClient(app)
    with client.websocket_connect("/ws/stream-affect") as ws:
        ws.send_text("I am delightfully proud of this work!")
        resp = ws.receive_json()
        assert resp["dominant_emotion"] == "joy"
        assert resp["valence"] > 0.0
        assert "confidence" in resp


def test_ws_client_stream_speech_audio_flow():
    client = TestClient(app)
    # Generate 0.2s synthetic 220Hz PCM sine wave
    t = np.linspace(0, 0.2, 3200, endpoint=False)
    pcm = (np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16).tobytes()
    b64_audio = base64.b64encode(pcm).decode("utf-8")

    with client.websocket_connect("/ws/stream-speech") as ws:
        ws.send_text(json.dumps({
            "audio_base64": b64_audio,
            "sample_rate": 16000,
        }))
        resp = ws.receive_json()
        assert "transcript" in resp
        assert "tokens" in resp
