"""Longitudinal Trajectory and Cohort Analytics Router."""

from typing import Optional
from fastapi import APIRouter, HTTPException

from src.api.deps import db
from src.analytics import LongitudinalProfileAnalyzer, generate_synthetic_cohort_benchmarks

router = APIRouter(tags=["Longitudinal Analytics"])


@router.get("/api/longitudinal/subjects")
async def list_longitudinal_subjects():
    """Returns list of distinct evaluated subjects, session counts, and date ranges."""
    try:
        subjects = db.list_distinct_subjects()
        return {
            "status": "success",
            "count": len(subjects),
            "subjects": subjects,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list subjects: {str(exc)}")


@router.get("/api/longitudinal/{subject_id}")
async def get_subject_trajectory(subject_id: str, cohort: Optional[str] = None):
    """Retrieves full longitudinal session trajectory and drift metrics for a subject."""
    try:
        points = db.get_subject_longitudinal_points(subject_id)
        benchmarks = generate_synthetic_cohort_benchmarks()
        cohort_obj = benchmarks.get(cohort) if cohort else None

        analyzer = LongitudinalProfileAnalyzer()
        profile = analyzer.analyze_profile(
            subject_id=subject_id,
            subject_name=subject_id,
            history_points=points,
            cohort=cohort_obj,
        )
        return {
            "status": "success",
            "profile": profile.to_dict(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Longitudinal analysis failed: {str(exc)}")


@router.get("/api/cohort/benchmarks")
async def get_cohort_benchmarks():
    """Returns population normative cohort benchmarks for comparative evaluation."""
    try:
        benchmarks = generate_synthetic_cohort_benchmarks()
        return {
            "status": "success",
            "benchmarks": {k: b.to_dict() for k, b in benchmarks.items()},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch benchmarks: {str(exc)}")
