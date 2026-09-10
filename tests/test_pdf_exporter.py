"""Unit tests for ClinicalPDFExporter and ReportLab PDF document compilation."""


from src.core.types import SessionRecord
from src.storage.models import AssessmentType, SessionMetadata
from src.utils.pdf_exporter import ClinicalPDFExporter


def test_pdf_exporter_availability():
    """Verifies that ReportLab is installed and available."""
    assert ClinicalPDFExporter.is_available() is True


def test_generate_pdf_standard_session():
    """Verifies standard session record compiles into valid PDF bytes."""
    now = 1700000000.0
    record = SessionRecord(
        session_id="pdf_test_session_001",
        start_time=now,
        end_time=now + 180.0,
        samples_count=250,
        average_affect={"valence": 0.42, "arousal": 0.65, "dominance": 0.35},
        dominant_emotion_distribution={"joy": 0.6, "surprise": 0.25, "neutral": 0.15},
        average_engagement=0.82,
        average_fatigue=0.18,
        average_attention=0.91,
        key_moments=[
            {"relative_time_sec": 12.5, "dominant_emotion": "joy", "confidence": 0.94, "valence": 0.8, "arousal": 0.7},
            {"relative_time_sec": 45.0, "dominant_emotion": "surprise", "confidence": 0.88, "valence": 0.5, "arousal": 0.9},
        ],
    )

    metadata = SessionMetadata(
        session_id="pdf_test_session_001",
        subject_id="PAT-9901",
        subject_name="Elena Rostova",
        assessment_type=AssessmentType.CLINICAL_SCREENING.value,
        evaluator="Dr. H. Vance",
        notes="Patient was relaxed, highly communicative, and responsive throughout the evaluation.",
        tags=["initial-assessment", "outpatient"],
    )

    anomalies = [
        {
            "timestamp": now + 45.0,
            "anomaly_type": "HYPER_AROUSAL_SPIKE",
            "severity": "MODERATE",
            "description": "Transient arousal surge corresponding to startling topic shift.",
            "recommended_action": "Note trigger phrase and re-evaluate at follow-up.",
            "metric_value": 0.90,
        }
    ]

    pdf_bytes = ClinicalPDFExporter.generate_pdf(record, anomalies=anomalies, metadata=metadata)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    # Check PDF magic bytes
    assert pdf_bytes.startswith(b"%PDF-")


def test_save_pdf_to_disk(tmp_path):
    """Verifies saving clinical PDF to filesystem path."""
    session_data = {
        "session_id": "file_export_sess",
        "start_time": 1000.0,
        "end_time": 1060.0,
        "samples_count": 60,
        "average_valence": 0.15,
        "average_arousal": 0.20,
        "average_dominance": 0.0,
        "dominant_emotion_distribution": {"neutral": 1.0},
    }

    out_file = tmp_path / "reports" / "diagnostic.pdf"
    res_path = ClinicalPDFExporter.save_pdf(session_data, out_file)

    assert res_path.exists()
    assert res_path.stat().st_size > 1000
    with open(res_path, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-"


def test_generate_pdf_empty_session():
    """Verifies that an empty or minimal session generates a valid PDF without exceptions."""
    minimal_record = SessionRecord(
        session_id="empty_sess",
        start_time=1000.0,
        end_time=1001.0,
        samples_count=0,
    )
    pdf_bytes = ClinicalPDFExporter.generate_pdf(minimal_record)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 1000
