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


def test_detect_anomalies_endpoint(client):
    payload = {
        "frames": [
            {
                "timestamp": 10.0,
                "dominant_emotion": "joy",
                "confidence": 0.88,
                "affect": {"valence": 0.7, "arousal": 0.6, "dominance": 0.5},
                "engagement_index": 0.8,
                "fatigue_level": 0.2,
                "attention_score": 0.85
            },
            {
                "timestamp": 11.0,
                "dominant_emotion": "anger",
                "confidence": 0.95,
                "affect": {"valence": -0.85, "arousal": 0.9, "dominance": 0.8},
                "engagement_index": 0.4,
                "fatigue_level": 0.75,
                "attention_score": 0.4
            }
        ]
    }
    response = client.post("/api/v1/detect-anomalies", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "total_anomalies" in data["data"]
    assert "events" in data["data"]
    assert len(data["data"]["events"]) > 0


def test_generate_diagnostic_report_endpoint(client):
    payload = {
        "session_id": "test_api_session",
        "frames": [
            {
                "timestamp": 1.0,
                "dominant_emotion": "neutral",
                "confidence": 0.7,
                "affect": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                "engagement_index": 0.6,
                "fatigue_level": 0.3,
                "attention_score": 0.7
            },
            {
                "timestamp": 2.0,
                "dominant_emotion": "joy",
                "confidence": 0.9,
                "affect": {"valence": 0.8, "arousal": 0.7, "dominance": 0.6},
                "engagement_index": 0.85,
                "fatigue_level": 0.2,
                "attention_score": 0.9
            }
        ]
    }
    response = client.post("/api/v1/generate-diagnostic-report", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["session_id"] == "test_api_session"
    assert "html_report" in data
    assert "markdown_report" in data
    assert "<!DOCTYPE html>" in data["html_report"]
    assert "Session Diagnostic Report" in data["markdown_report"]


def test_websocket_speech_stream(client):
    with client.websocket_connect("/ws/stream-speech") as websocket:
        websocket.send_text("I am so excited and overjoyed with this release! 🎉")
        data = websocket.receive_json()
        assert "transcript" in data
        assert data["dominant_emotion"] == "joy"
        assert data["valence"] > 0.0
        assert "tokens" in data
        assert len(data["tokens"]) > 0


def test_websocket_speech_stream_json_payloads(client):
    import json
    import base64
    import numpy as np

    with client.websocket_connect("/ws/stream-speech") as websocket:
        # 1. Test JSON text payload
        websocket.send_text(json.dumps({"text": "I feel very anxious and scared today."}))
        data = websocket.receive_json()
        assert "transcript" in data
        assert data["dominant_emotion"] in ["fear", "sadness", "neutral"]

        # 2. Test JSON audio_base64 payload
        t = np.linspace(0, 0.5, 8000, dtype=np.float32)
        sine_pcm = (np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16).tobytes()
        b64_audio = base64.b64encode(sine_pcm).decode("utf-8")
        websocket.send_text(json.dumps({"audio_base64": b64_audio, "sample_rate": 16000}))
        data2 = websocket.receive_json()
        assert "transcript" in data2
        assert "tokens" in data2


def test_api_session_crud_and_stats(client):
    """Tests the full REST persistence lifecycle: creation, listing, metadata patch, stats, and deletion."""
    session_id = "test_api_crud_sess_01"
    payload = {
        "session_id": session_id,
        "start_time": 1000.0,
        "end_time": 1050.0,
        "samples_count": 10,
        "timeline": [
            {
                "timestamp": 1000.0,
                "dominant_emotion": "joy",
                "confidence": 0.9,
                "affect": {"valence": 0.7, "arousal": 0.6, "dominance": 0.5},
            }
        ],
        "metadata": {
            "subject_name": "API Tester",
            "assessment_type": "clinical_screening",
            "notes": "Testing REST persistence",
            "tags": ["ci", "rest"],
        },
    }

    # 1. Create Session
    create_res = client.post("/api/sessions", json=payload)
    assert create_res.status_code == 200
    assert create_res.json()["session_id"] == session_id

    # 2. List Sessions
    list_res = client.get("/api/sessions?assessment_type=clinical_screening")
    assert list_res.status_code == 200
    sessions = list_res.json()["sessions"]
    matching = [s for s in sessions if s["session_id"] == session_id]
    assert len(matching) == 1
    assert matching[0]["subject_name"] == "API Tester"

    # 3. Get Session
    get_res = client.get(f"/api/sessions/{session_id}")
    assert get_res.status_code == 200
    sess_data = get_res.json()
    assert sess_data["session_id"] == session_id
    assert len(sess_data["timeline"]) == 1

    # 4. Patch Metadata
    patch_res = client.patch(
        f"/api/sessions/{session_id}/metadata",
        json={"notes": "Updated through PATCH", "tags": ["ci", "rest", "verified"]}
    )
    assert patch_res.status_code == 200

    get_updated = client.get(f"/api/sessions/{session_id}")
    assert get_updated.json()["metadata"]["notes"] == "Updated through PATCH"
    assert "verified" in get_updated.json()["metadata"]["tags"]

    # 5. Platform Stats
    stats_res = client.get("/api/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()["stats"]
    assert stats["total_sessions"] >= 1

    # 6. Delete Session
    del_res = client.delete(f"/api/sessions/{session_id}")
    assert del_res.status_code == 200

    # 7. Verify 404
    get_404 = client.get(f"/api/sessions/{session_id}")
    assert get_404.status_code == 404

    # 8. Migrate Endpoint
    migrate_res = client.post("/api/sessions/migrate")
    assert migrate_res.status_code == 200
    assert "migrated_sessions_count" in migrate_res.json()





