"""Integration tests for FastAPI REST Microservice endpoints."""

import pytest
from starlette.testclient import TestClient
from server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"
    assert "active_mode" in data


def test_predict_single_text_endpoint(client):
    payload = {
        "text": "I am thrilled and so proud of our launch! 🎉",
        "mode": "lexical"
    }
    response = client.post("/api/v1/predict-text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["dominant_emotion"] == "joy"
    assert data["data"]["affect"]["valence"] > 0.0


def test_predict_single_text_empty_error(client):
    payload = {"text": "   "}
    response = client.post("/api/v1/predict-text", json=payload)
    assert response.status_code == 400


def test_analyze_dialogue_endpoint(client):
    payload = {
        "transcript": """[10:00 AM] Alex: I'm really stressed about the demo!
[10:01 AM] Jordan: Don't worry, we tested everything and it works great!"""
    }
    response = client.post("/api/v1/analyze-dialogue", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["total_turns"] == 2
    assert len(data["data"]["speakers"]) == 2
    assert "Alex" in data["data"]["speakers"]


def test_batch_predict_endpoint(client):
    payload = {
        "messages": [
            "I love this product! ❤️",
            "This service is completely broken.",
            "Normal status report."
        ]
    }
    response = client.post("/api/v1/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_messages"] == 3
    assert "distribution" in data
    assert len(data["results"]) == 3


def test_websocket_affect_stream(client):
    with client.websocket_connect("/ws/stream-affect") as websocket:
        websocket.send_text("I am so happy to see you! 🎉")
        data = websocket.receive_json()
        assert data["dominant_emotion"] == "joy"
        assert data["valence"] > 0.0
        assert "empathy_advice" in data

