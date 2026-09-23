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


def test_api_analyze_dyadic_endpoint(client):
    payload = {
        "participant_a_samples": [
            {"timestamp": float(i), "valence": 0.5 + 0.05 * i, "arousal": 0.4, "smile": 0.8, "yaw": 2.0}
            for i in range(10)
        ],
        "participant_b_samples": [
            {"timestamp": float(i), "valence": 0.55 + 0.04 * i, "arousal": 0.45, "smile": 0.75, "yaw": -2.0}
            for i in range(10)
        ],
        "diarization": {
            "speakers": ["Speaker_0", "Speaker_1"],
            "speaker_durations": {"Speaker_0": 5.0, "Speaker_1": 5.0},
            "dominance_ratios": {"Speaker_0": 0.5, "Speaker_1": 0.5},
            "interruption_count": 0,
            "total_speech_duration": 10.0,
            "total_audio_duration": 10.0,
            "turns": [
                {"speaker_id": "Speaker_0", "start_time": 0.0, "end_time": 5.0, "duration": 5.0},
                {"speaker_id": "Speaker_1", "start_time": 5.0, "end_time": 10.0, "duration": 5.0},
            ],
        },
    }
    response = client.post("/api/analyze/dyadic", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "metrics" in data
    assert data["metrics"]["rapport_score"] >= 70.0
    assert data["metrics"]["conversational_balance"] == 1.0


def test_api_audio_diarize_endpoint(client):
    import base64
    import io
    import numpy as np
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV")
    b64_audio = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "audio_base64": b64_audio,
        "sample_rate": 16000,
        "num_speakers": 1,
    }
    response = client.post("/api/audio/diarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "diarization" in data
    assert len(data["diarization"]["speakers"]) == 1


def test_longitudinal_endpoints(client):
    """Tests Phase 7 longitudinal subject trajectory and cohort benchmark REST endpoints."""
    # 1. Test Cohort Benchmarks
    bench_res = client.get("/api/cohort/benchmarks")
    assert bench_res.status_code == 200
    b_data = bench_res.json()
    assert b_data["status"] == "success"
    assert "Clinical Screening" in b_data["benchmarks"]
    assert "Talent Interview" in b_data["benchmarks"]

    # 2. Test Subjects List
    subj_res = client.get("/api/longitudinal/subjects")
    assert subj_res.status_code == 200
    s_data = subj_res.json()
    assert s_data["status"] == "success"
    assert "subjects" in s_data

    # 3. Test Trajectory Query
    traj_res = client.get("/api/longitudinal/SUBJ_NONEXISTENT?cohort=Clinical%20Screening")
    assert traj_res.status_code == 200
    t_data = traj_res.json()
    assert t_data["status"] == "success"
    assert t_data["profile"]["subject_id"] == "SUBJ_NONEXISTENT"
    assert t_data["profile"]["total_sessions"] == 0


def test_api_edge_endpoints(client):
    # 1. Test Hardware Profile Endpoint
    hw_res = client.get("/api/edge/hardware")
    assert hw_res.status_code == 200
    hw_data = hw_res.json()
    assert hw_data["status"] == "success"
    assert "profile" in hw_data
    assert "provider" in hw_data["profile"]

    # 2. Test Benchmark Endpoint
    bench_res = client.post(
        "/api/edge/benchmark",
        json={
            "model_name": "vision_mesh",
            "precision": "INT8",
            "iterations": 10,
        }
    )
    assert bench_res.status_code == 200
    b_data = bench_res.json()
    assert b_data["status"] == "success"
    assert b_data["benchmark"]["model_name"] == "vision_mesh"
    assert b_data["benchmark"]["mean_latency_ms"] > 0.0

    # 3. Test Quantization Estimation Endpoint
    quant_res = client.post(
        "/api/edge/quantize",
        json={
            "model_name": "audio_ser",
            "target_precision": "INT8",
        }
    )
    assert quant_res.status_code == 200
    q_data = quant_res.json()
    assert q_data["status"] == "success"
    assert q_data["quantization"]["compression_ratio"] > 1.0
    assert q_data["quantization"]["quantized_size_mb"] < q_data["quantization"]["original_size_mb"]


def test_agent_api_endpoints(client):
    """Test Phase 9 agentic endpoints: providers, diagnostic synthesis, and chat copilot."""
    # 1. Providers endpoint
    p_res = client.get("/api/agent/providers")
    assert p_res.status_code == 200
    p_data = p_res.json()
    assert p_data["status"] == "success"
    assert "rule_based" in p_data["providers"]
    assert "ollama" in p_data["providers"]

    # 2. Synthesize endpoint
    synth_res = client.post(
        "/api/agent/synthesize",
        json={
            "session_id": "sess_api_test_01",
            "subject_id": "TestSubject",
            "assessment_type": "Clinical Screening",
            "timeline_samples": [
                {"valence": 0.2, "arousal": 0.1, "timestamp_sec": 1.0},
                {"valence": -0.5, "arousal": 0.7, "timestamp_sec": 2.0},
            ],
            "anomalies": [{"timestamp_sec": 2.0, "anomaly_type": "valence_crash"}],
            "provider_name": "rule_based",
        }
    )
    assert synth_res.status_code == 200
    s_data = synth_res.json()
    assert s_data["status"] == "success"
    assert s_data["synthesis"]["session_id"] == "sess_api_test_01"
    assert "risk_assessment" in s_data["synthesis"]

    # 3. Chat copilot endpoint
    chat_res = client.post(
        "/api/agent/chat",
        json={
            "session_id": "sess_api_test_01",
            "question": "Why did the candidate display elevated stress?",
            "provider_name": "rule_based",
        }
    )
    assert chat_res.status_code == 200
    c_data = chat_res.json()
    assert c_data["status"] == "success"
    assert len(c_data["response"]["answer"]) > 10
    assert len(c_data["response"]["suggested_followups"]) > 0


def test_biometric_api_endpoints(client):
    """Test Phase 10 remote biometric REST endpoints."""
    # 1. Status endpoint
    status_res = client.get("/api/biometrics/status")
    assert status_res.status_code == 200
    st_data = status_res.json()
    assert st_data["status"] == "online"
    assert "Plane-Orthogonal-to-Skin (POS)" in st_data["algorithms"]
    assert "RMSSD" in st_data["hrv_metrics"]

    # 2. Stress computation endpoint
    stress_res = client.post(
        "/api/biometrics/stress",
        json={
            "bpm": 82.0,
            "rmssd_ms": 32.0,
            "baevsky_si": 110.0,
            "rpm": 16.0,
            "valence": -0.2,
            "arousal": 0.4,
            "vocal_jitter": 0.02,
        }
    )
    assert stress_res.status_code == 200
    s_data = stress_res.json()
    assert s_data["status"] == "success"
    assert "stress" in s_data
    assert 0.0 <= s_data["stress"]["stress_index"] <= 1.0
    assert "classification" in s_data["stress"]
    assert "sympathetic_tone" in s_data["stress"]

    # 3. rPPG extraction endpoint
    from src.analytics.biometrics import BiometricEngine
    _, rgb_matrix = BiometricEngine.generate_synthetic_bvp_stream(
        duration_sec=3.0, fps=30.0, bpm=72.0, noise_level=0.01
    )
    rppg_res = client.post(
        "/api/biometrics/rppg",
        json={
            "rgb_samples": rgb_matrix.tolist(),
            "fps": 30.0,
            "valence": 0.1,
            "arousal": 0.2,
        }
    )
    assert rppg_res.status_code == 200
    r_data = rppg_res.json()
    assert r_data["status"] == "success"
    telem = r_data["telemetry"]
    assert "pulse" in telem
    assert "hrv" in telem
    assert "respiration" in telem
    assert "autonomic_stress" in telem
    assert 45.0 <= telem["pulse"]["bpm"] <= 180.0







