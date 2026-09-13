"""Unit and integration tests for Dyadic Live Studio processing and persistence."""

import numpy as np
from src.ui.video_processor import MultimodalStreamContext
from src.core.types import (
    MultiFaceResult,
    TrackedFace,
    VisionEmotionResult,
    DiarizationResult,
    SpeakerTurn,
    DyadicInteractionMetrics,
)
from src.fusion.interaction_dynamics import DyadicInteractionAnalyzer
from src.storage import SessionDatabase, SessionMetadata, AssessmentType
from src.core.types import SessionRecord


def test_stream_context_multi_face_history():
    """Verify MultimodalStreamContext records and retrieves multi-face tracking frames."""
    ctx = MultimodalStreamContext(window_size=15)
    assert ctx.get_latest_multi_face() is None
    assert len(ctx.get_multi_face_history()) == 0

    vis_dummy = VisionEmotionResult(face_detected=True)
    # Add 5 frames
    for i in range(5):
        mf = MultiFaceResult(
            timestamp=float(i),
            faces=[
                TrackedFace(track_id=0, bbox=(10, 10, 50, 50), centroid=(35.0, 35.0), vision_result=vis_dummy),
                TrackedFace(track_id=1, bbox=(70, 10, 110, 50), centroid=(90.0, 35.0), vision_result=vis_dummy),
            ],
            face_count=2,
        )
        ctx.update_multi_face(mf)

    assert ctx.get_latest_multi_face() is not None
    assert ctx.get_latest_multi_face().face_count == 2
    history = ctx.get_multi_face_history()
    assert len(history) == 5
    assert history[-1].timestamp == 4.0


def test_dyadic_simulation_flow():
    """Verify dyadic interaction analysis produces valid metrics for two participants."""
    analyzer = DyadicInteractionAnalyzer()

    samples_a = []
    samples_b = []

    for t in range(20):
        val_a = 0.6 + 0.1 * np.sin(t / 2.0)
        val_b = 0.5 + 0.1 * np.sin(t / 2.0)  # In-phase synchrony

        samples_a.append({
            "timestamp": float(t),
            "valence": val_a,
            "arousal": 0.4,
            "smile": 0.8,
            "yaw": 0.0,
            "dominant_emotion": "joy",
            "confidence": 0.85,
        })
        samples_b.append({
            "timestamp": float(t),
            "valence": val_b,
            "arousal": 0.35,
            "smile": 0.75,
            "yaw": 0.0,
            "dominant_emotion": "joy",
            "confidence": 0.80,
        })

    diar = DiarizationResult(
        turns=[
            SpeakerTurn("Speaker_0", 0.0, 10.0, 10.0, "Turn 1"),
            SpeakerTurn("Speaker_1", 10.0, 20.0, 10.0, "Turn 2"),
        ],
        speakers=["Speaker_0", "Speaker_1"],
        speaker_durations={"Speaker_0": 10.0, "Speaker_1": 10.0},
        dominance_ratios={"Speaker_0": 0.5, "Speaker_1": 0.5},
        interruption_count=1,
        total_speech_duration=20.0,
        total_audio_duration=20.0,
    )

    metrics = analyzer.analyze(samples_a, samples_b, diar)

    assert isinstance(metrics, DyadicInteractionMetrics)
    assert 0.0 <= metrics.rapport_score <= 100.0
    assert metrics.valence_synchrony > 0.5  # High synchrony
    assert "Resonance" in metrics.resonance_category or "Collaborative" in metrics.resonance_category
    assert 0.0 <= metrics.conversational_balance <= 1.0


def test_dyadic_sqlite_persistence(tmp_path):
    """Verify dyadic session with metadata persists and retrieves intact through SessionDatabase."""
    db_file = tmp_path / "test_dyadic.db"
    db = SessionDatabase(db_path=db_file)

    now = 1700000000.0
    record = SessionRecord(
        session_id="sess_dyadic_101",
        start_time=now,
        end_time=now + 30.0,
        samples_count=50,
        average_affect={"valence": 0.5, "arousal": 0.4, "dominance": 0.5},
        dominant_emotion_distribution={"joy": 0.9, "neutral": 0.1},
    )

    metadata = SessionMetadata(
        session_id="sess_dyadic_101",
        subject_name="Alex & Jordan",
        assessment_type=AssessmentType.DYADIC_INTERVIEW.value,
        evaluator="Dr. Marcus",
        notes="Dyadic interaction assessment with 84.5 rapport index.",
        tags=["dyadic", "interpersonal", "high-resonance"],
    )

    db.save_session(record, metadata=metadata)

    loaded = db.get_session("sess_dyadic_101")
    assert loaded is not None
    assert loaded.session_id == "sess_dyadic_101"
    assert loaded.metadata.assessment_type == AssessmentType.DYADIC_INTERVIEW.value
    assert loaded.metadata.subject_name == "Alex & Jordan"
    assert "dyadic" in loaded.metadata.tags


def test_dyadic_empty_samples_fallback():
    """Verify analyzer gracefully handles empty or mismatched sample lengths."""
    analyzer = DyadicInteractionAnalyzer()
    metrics = analyzer.analyze([], [])
    assert metrics.rapport_score == 50.0
    assert metrics.valence_synchrony == 0.0
    assert metrics.resonance_category == "Insufficient Data"
