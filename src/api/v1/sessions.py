"""Session Persistence and Analytics Router."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from src.api.deps import db
from src.api.v1.schemas import SaveSessionRequest, MetadataUpdateRequest

router = APIRouter(tags=["Session Persistence"])


@router.get("/api/sessions")
def list_persisted_sessions(
    assessment_type: Optional[str] = Query(None, description="Filter by assessment type"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    search: Optional[str] = Query(None, description="Search term across session ID, subject name, or notes"),
    limit: int = Query(50, ge=1, le=200, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """Lists saved multimodal sessions with filtering, pagination, and metadata."""
    sessions = db.list_sessions(
        assessment_type=assessment_type,
        tag=tag,
        search_query=search,
        limit=limit,
        offset=offset
    )
    return {
        "status": "success",
        "count": len(sessions),
        "limit": limit,
        "offset": offset,
        "sessions": sessions,
    }


@router.get("/api/sessions/{session_id}")
def get_persisted_session(session_id: str, include_samples: bool = Query(True, description="Whether to include granular timeline samples")):
    """Retrieves full persisted session, aggregates, metadata, and anomaly alerts."""
    session = db.get_session(session_id, include_samples=include_samples)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session.to_dict()


@router.post("/api/sessions")
def save_persisted_session(request: SaveSessionRequest):
    """Creates or updates a session in SQLite database."""
    sess_dict = request.model_dump()
    meta_dict = sess_dict.pop("metadata", None)
    anom_list = sess_dict.pop("anomalies", None)
    saved_id = db.save_session(sess_dict, metadata=meta_dict, anomalies=anom_list)
    return {"status": "success", "session_id": saved_id}


@router.patch("/api/sessions/{session_id}/metadata")
def update_session_metadata(session_id: str, request: MetadataUpdateRequest):
    """Updates candidate/patient metadata, notes, and tags for a session."""
    existing = db.get_session(session_id, include_samples=False)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    meta_dict = request.model_dump(exclude_unset=True)
    meta_dict["session_id"] = session_id
    success = db.update_metadata(session_id, meta_dict)
    return {"status": "success" if success else "failed", "session_id": session_id}


@router.delete("/api/sessions/{session_id}")
def delete_persisted_session(session_id: str):
    """Deletes a session and cascading samples and anomalies from database."""
    deleted = db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "success", "deleted_session_id": session_id}


@router.post("/api/sessions/migrate")
def trigger_json_migration():
    """Migrates all legacy flat JSON sessions into SQLite database."""
    count = db.migrate_from_json()
    return {"status": "success", "migrated_sessions_count": count}


@router.get("/api/stats")
def get_platform_statistics():
    """Returns platform-wide metrics: total sessions, samples, anomalies, and assessment types."""
    return {"status": "success", "stats": db.get_stats()}
