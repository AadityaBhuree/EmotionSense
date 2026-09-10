"""Tests for SQLite session persistence, metadata tracking, and JSON migration."""

import json
import time
import pytest

from src.core.types import SessionRecord
from src.storage import SessionDatabase, SessionMetadata, AssessmentType


@pytest.fixture
def temp_db(tmp_path):
    """Creates a temporary SQLite database for testing."""
    db_file = tmp_path / "test_emotionsense.db"
    return SessionDatabase(db_path=db_file)


def test_database_initialization_and_tables(temp_db):
    """Verifies SQLite tables and schema are created properly."""
    assert temp_db.db_path.exists()
    with temp_db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "sessions" in tables
        assert "session_metadata" in tables
        assert "session_samples" in tables
        assert "session_anomalies" in tables


def test_save_and_retrieve_session(temp_db):
    """Tests saving a full session with samples and retrieving it."""
    session_id = "test_sess_001"
    now = time.time()
    timeline = [
        {
            "timestamp": now,
            "dominant_emotion": "joy",
            "confidence": 0.95,
            "affect": {"valence": 0.7, "arousal": 0.6, "dominance": 0.5},
            "engagement_index": 0.85,
            "fatigue_level": 0.15,
            "attention_score": 0.90,
            "quadrant": "High Valence / High Arousal (Exuberant)",
            "probabilities": {"joy": 0.95, "neutral": 0.05},
            "text": "Thrilled with the progress!",
        },
        {
            "timestamp": now + 1.0,
            "dominant_emotion": "surprise",
            "confidence": 0.88,
            "affect": {"valence": 0.4, "arousal": 0.8, "dominance": 0.4},
            "engagement_index": 0.90,
            "fatigue_level": 0.20,
            "attention_score": 0.85,
            "quadrant": "High Valence / High Arousal",
            "probabilities": {"surprise": 0.88, "joy": 0.12},
            "text": None,
        }
    ]

    record = SessionRecord(
        session_id=session_id,
        start_time=now,
        end_time=now + 5.0,
        samples_count=2,
        timeline=timeline,
        average_affect={"valence": 0.55, "arousal": 0.70, "dominance": 0.45},
        dominant_emotion_distribution={"joy": 0.5, "surprise": 0.5},
        average_engagement=0.875,
        average_fatigue=0.175,
        average_attention=0.875,
        key_moments=[{"relative_time_sec": 1.0, "dominant_emotion": "surprise", "confidence": 0.88, "valence": 0.4, "arousal": 0.8}],
    )

    metadata = SessionMetadata(
        session_id=session_id,
        subject_id="SUBJ-42",
        subject_name="Dr. Alex Rivera",
        assessment_type=AssessmentType.CLINICAL_SCREENING.value,
        evaluator="Clinician Marcus",
        notes="Patient displays high responsiveness.",
        tags=["baseline", "intake"],
    )

    anomalies = [
        {
            "timestamp": now + 1.0,
            "anomaly_type": "HYPER_AROUSAL_SPIKE",
            "severity": "MODERATE",
            "description": "Sudden arousal transition exceeding normal threshold.",
            "recommended_action": "Verify acoustic audio clarity.",
            "metric_value": 0.80,
        }
    ]

    saved_id = temp_db.save_session(record, metadata=metadata, anomalies=anomalies)
    assert saved_id == session_id

    # Retrieve and verify
    retrieved = temp_db.get_session(session_id, include_samples=True)
    assert retrieved is not None
    assert retrieved.session_id == session_id
    assert retrieved.samples_count == 2
    assert retrieved.duration_seconds == 5.0
    assert retrieved.average_valence == 0.55
    assert retrieved.metadata.subject_name == "Dr. Alex Rivera"
    assert retrieved.metadata.assessment_type == AssessmentType.CLINICAL_SCREENING.value
    assert len(retrieved.timeline) == 2
    assert retrieved.timeline[0]["dominant_emotion"] == "joy"
    assert len(retrieved.anomalies) == 1
    assert retrieved.anomalies[0]["anomaly_type"] == "HYPER_AROUSAL_SPIKE"


def test_list_sessions_with_filters(temp_db):
    """Tests listing sessions with type, tag, and search query filters."""
    now = time.time()
    # Create 3 distinct sessions
    s1 = {"session_id": "sess_interview_01", "start_time": now - 100, "samples_count": 10}
    m1 = {"assessment_type": AssessmentType.TALENT_INTERVIEW.value, "subject_name": "Jordan Smith", "tags": ["lead-eng", "tech"]}
    temp_db.save_session(s1, metadata=m1)

    s2 = {"session_id": "sess_clinical_01", "start_time": now - 50, "samples_count": 15}
    m2 = {"assessment_type": AssessmentType.CLINICAL_SCREENING.value, "subject_name": "Sarah Connor", "tags": ["post-op", "trauma"]}
    temp_db.save_session(s2, metadata=m2)

    s3 = {"session_id": "sess_wellness_01", "start_time": now, "samples_count": 20}
    m3 = {"assessment_type": AssessmentType.WELLNESS_TRACKING.value, "subject_name": "Taylor Doe", "tags": ["stress"]}
    temp_db.save_session(s3, metadata=m3)

    # Filter by assessment type
    interviews = temp_db.list_sessions(assessment_type=AssessmentType.TALENT_INTERVIEW.value)
    assert len(interviews) == 1
    assert interviews[0]["session_id"] == "sess_interview_01"

    # Filter by tag
    tech_sessions = temp_db.list_sessions(tag="tech")
    assert len(tech_sessions) == 1
    assert tech_sessions[0]["session_id"] == "sess_interview_01"

    # Search by subject name
    sarah_sessions = temp_db.list_sessions(search_query="Sarah")
    assert len(sarah_sessions) == 1
    assert sarah_sessions[0]["session_id"] == "sess_clinical_01"


def test_update_session_metadata(temp_db):
    """Tests updating session metadata."""
    sess_id = "sess_update_test"
    temp_db.save_session({"session_id": sess_id, "start_time": time.time()})

    new_meta = SessionMetadata(
        session_id=sess_id,
        subject_name="Morgan Chase",
        assessment_type=AssessmentType.RESEARCH_STUDY.value,
        notes="Updated participant protocol",
        tags=["cohort_b"]
    )
    success = temp_db.update_metadata(sess_id, new_meta)
    assert success is True

    stored = temp_db.get_session(sess_id, include_samples=False)
    assert stored is not None
    assert stored.metadata.subject_name == "Morgan Chase"
    assert stored.metadata.assessment_type == AssessmentType.RESEARCH_STUDY.value
    assert "cohort_b" in stored.metadata.tags


def test_session_cascade_delete(temp_db):
    """Tests that deleting a session cascades to metadata, samples, and anomalies."""
    sess_id = "sess_delete_me"
    now = time.time()
    timeline = [{"timestamp": now, "dominant_emotion": "anger", "confidence": 0.9, "affect": {"valence": -0.5, "arousal": 0.7, "dominance": 0.6}}]
    anomalies = [{"timestamp": now, "anomaly_type": "VALENCE_CRASH", "severity": "HIGH", "description": "Crash"}]
    
    temp_db.save_session(
        {"session_id": sess_id, "start_time": now, "timeline": timeline},
        metadata={"subject_name": "Ghost User"},
        anomalies=anomalies
    )

    assert temp_db.get_session(sess_id) is not None
    deleted = temp_db.delete_session(sess_id)
    assert deleted is True
    assert temp_db.get_session(sess_id) is None

    # Check child tables are empty for this session
    with temp_db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM session_samples WHERE session_id = ?", (sess_id,))
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT COUNT(*) FROM session_metadata WHERE session_id = ?", (sess_id,))
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT COUNT(*) FROM session_anomalies WHERE session_id = ?", (sess_id,))
        assert cursor.fetchone()[0] == 0


def test_json_migration_idempotent(temp_db, tmp_path):
    """Tests migrating flat JSON session files to SQLite."""
    json_dir = tmp_path / "sessions_json"
    json_dir.mkdir()

    # Create dummy JSON session files
    for i in range(3):
        data = {
            "session_id": f"legacy_session_{i}",
            "start_time": 1000.0 + i * 10,
            "end_time": 1020.0 + i * 10,
            "samples_count": 5,
            "average_affect": {"valence": 0.2, "arousal": 0.3, "dominance": 0.1},
            "dominant_emotion_distribution": {"neutral": 0.8, "joy": 0.2},
            "timeline": []
        }
        with open(json_dir / f"legacy_session_{i}.json", "w") as f:
            json.dump(data, f)

    count = temp_db.migrate_from_json(json_dir)
    assert count == 3

    # Idempotent re-run
    count2 = temp_db.migrate_from_json(json_dir)
    assert count2 == 3
    assert len(temp_db.list_sessions()) == 3


def test_database_stats(temp_db):
    """Tests calculation of platform database statistics."""
    now = time.time()
    temp_db.save_session(
        {"session_id": "stat_1", "start_time": now, "end_time": now + 60, "samples_count": 100},
        metadata={"assessment_type": AssessmentType.CLINICAL_SCREENING.value}
    )
    temp_db.save_session(
        {"session_id": "stat_2", "start_time": now, "end_time": now + 120, "samples_count": 200},
        metadata={"assessment_type": AssessmentType.CLINICAL_SCREENING.value}
    )

    stats = temp_db.get_stats()
    assert stats["total_sessions"] == 2
    assert stats["total_samples"] == 300
    assert stats["average_duration_seconds"] == 90.0
    assert stats["assessment_types"][AssessmentType.CLINICAL_SCREENING.value] == 2
