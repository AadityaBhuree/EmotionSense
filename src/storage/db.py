"""SQLite persistence engine for EmotionSense session intelligence."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from config import DB_PATH, SESSIONS_DIR
from src.core.types import SessionRecord
from src.storage.models import AssessmentType, SessionMetadata, StoredSession
from src.utils.logger import logger


class SessionDatabase:
    """Enterprise SQLite database engine for multimodal sessions, metadata, and affective anomalies."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a connection with Row factory and performance pragmas."""
        conn = sqlite3.connect(str(self.db_path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        """Initializes database schema and indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    samples_count INTEGER DEFAULT 0,
                    duration_seconds REAL DEFAULT 0.0,
                    avg_valence REAL DEFAULT 0.0,
                    avg_arousal REAL DEFAULT 0.0,
                    avg_dominance REAL DEFAULT 0.0,
                    avg_engagement REAL DEFAULT 0.0,
                    avg_fatigue REAL DEFAULT 0.0,
                    avg_attention REAL DEFAULT 0.0,
                    dominant_emotion TEXT DEFAULT 'neutral',
                    emotion_distribution TEXT,
                    key_moments TEXT,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_metadata (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
                    subject_id TEXT,
                    subject_name TEXT,
                    assessment_type TEXT DEFAULT 'general_affect',
                    evaluator TEXT,
                    notes TEXT,
                    tags TEXT,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    timestamp REAL NOT NULL,
                    dominant_emotion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    valence REAL NOT NULL,
                    arousal REAL NOT NULL,
                    dominance REAL NOT NULL,
                    engagement_index REAL DEFAULT 0.0,
                    fatigue_level REAL DEFAULT 0.0,
                    attention_score REAL DEFAULT 0.0,
                    quadrant TEXT,
                    probabilities TEXT,
                    text_payload TEXT
                );

                CREATE TABLE IF NOT EXISTS session_anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    timestamp REAL NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    recommended_action TEXT,
                    metric_value REAL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);
                CREATE INDEX IF NOT EXISTS idx_sessions_dominant_emotion ON sessions(dominant_emotion);
                CREATE INDEX IF NOT EXISTS idx_metadata_type ON session_metadata(assessment_type);
                CREATE INDEX IF NOT EXISTS idx_samples_session_id ON session_samples(session_id);
                CREATE INDEX IF NOT EXISTS idx_anomalies_session_id ON session_anomalies(session_id);
            """)
            conn.commit()

    def save_session(
        self,
        session_record: Union[SessionRecord, Dict[str, Any]],
        metadata: Optional[Union[SessionMetadata, Dict[str, Any]]] = None,
        anomalies: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Saves or updates a complete session, including metadata, samples, and anomalies."""
        if isinstance(session_record, SessionRecord):
            rec_dict = session_record.to_dict()
        else:
            rec_dict = dict(session_record)

        session_id = rec_dict.get("session_id", f"session_{int(time.time())}")
        start_time = float(rec_dict.get("start_time", time.time()))
        end_time = rec_dict.get("end_time")
        if end_time is not None:
            end_time = float(end_time)
            duration_sec = max(0.0, end_time - start_time)
        else:
            duration_sec = 0.0

        samples_count = int(rec_dict.get("samples_count", 0))
        avg_affect = rec_dict.get("average_affect", {})
        avg_v = float(avg_affect.get("valence", 0.0))
        avg_a = float(avg_affect.get("arousal", 0.0))
        avg_d = float(avg_affect.get("dominance", 0.0))

        avg_eng = float(rec_dict.get("average_engagement", 0.0))
        avg_fat = float(rec_dict.get("average_fatigue", 0.0))
        avg_att = float(rec_dict.get("average_attention", 0.0))

        distribution = rec_dict.get("dominant_emotion_distribution", {})
        if distribution:
            dominant_emotion = max(distribution.items(), key=lambda x: x[1])[0]
        else:
            dominant_emotion = "neutral"

        emotion_dist_json = json.dumps(distribution)
        key_moments_json = json.dumps(rec_dict.get("key_moments", []))
        created_at = time.time()

        # Metadata extraction
        if metadata is None:
            meta_obj = SessionMetadata(session_id=session_id)
        elif isinstance(metadata, SessionMetadata):
            meta_obj = metadata
            meta_obj.session_id = session_id
        else:
            metadata_dict = dict(metadata)
            metadata_dict["session_id"] = session_id
            meta_obj = SessionMetadata.from_dict(metadata_dict)

        tags_json = json.dumps(meta_obj.tags)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Upsert into sessions table
            cursor.execute("""
                INSERT INTO sessions (
                    session_id, start_time, end_time, samples_count, duration_seconds,
                    avg_valence, avg_arousal, avg_dominance, avg_engagement, avg_fatigue,
                    avg_attention, dominant_emotion, emotion_distribution, key_moments, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    start_time=excluded.start_time,
                    end_time=excluded.end_time,
                    samples_count=excluded.samples_count,
                    duration_seconds=excluded.duration_seconds,
                    avg_valence=excluded.avg_valence,
                    avg_arousal=excluded.avg_arousal,
                    avg_dominance=excluded.avg_dominance,
                    avg_engagement=excluded.avg_engagement,
                    avg_fatigue=excluded.avg_fatigue,
                    avg_attention=excluded.avg_attention,
                    dominant_emotion=excluded.dominant_emotion,
                    emotion_distribution=excluded.emotion_distribution,
                    key_moments=excluded.key_moments
            """, (
                session_id, start_time, end_time, samples_count, duration_sec,
                avg_v, avg_a, avg_d, avg_eng, avg_fat,
                avg_att, dominant_emotion, emotion_dist_json, key_moments_json, created_at
            ))

            # Upsert into metadata table
            cursor.execute("""
                INSERT INTO session_metadata (
                    session_id, subject_id, subject_name, assessment_type,
                    evaluator, notes, tags, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    subject_id=excluded.subject_id,
                    subject_name=excluded.subject_name,
                    assessment_type=excluded.assessment_type,
                    evaluator=excluded.evaluator,
                    notes=excluded.notes,
                    tags=excluded.tags,
                    updated_at=excluded.updated_at
            """, (
                session_id, meta_obj.subject_id, meta_obj.subject_name,
                meta_obj.assessment_type, meta_obj.evaluator, meta_obj.notes,
                tags_json, time.time()
            ))

            # Store timeline samples if provided
            timeline = rec_dict.get("timeline", [])
            if timeline:
                cursor.execute("DELETE FROM session_samples WHERE session_id = ?", (session_id,))
                sample_rows = []
                for s in timeline:
                    aff = s.get("affect", {})
                    ts = float(s.get("timestamp", start_time))
                    dom_emo = s.get("dominant_emotion", "neutral")
                    conf = float(s.get("confidence", 0.0))
                    val = float(aff.get("valence", 0.0))
                    aro = float(aff.get("arousal", 0.0))
                    dom = float(aff.get("dominance", 0.0))
                    eng = float(s.get("engagement_index", 0.0))
                    fat = float(s.get("fatigue_level", 0.0))
                    att = float(s.get("attention_score", 0.0))
                    quad = s.get("quadrant", "")
                    probs = json.dumps(s.get("probabilities", {}))
                    txt = s.get("text")
                    txt_str = json.dumps(txt) if txt is not None else None
                    sample_rows.append((
                        session_id, ts, dom_emo, conf, val, aro, dom,
                        eng, fat, att, quad, probs, txt_str
                    ))

                cursor.executemany("""
                    INSERT INTO session_samples (
                        session_id, timestamp, dominant_emotion, confidence, valence,
                        arousal, dominance, engagement_index, fatigue_level, attention_score,
                        quadrant, probabilities, text_payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, sample_rows)

            # Store anomalies if provided
            if anomalies:
                cursor.execute("DELETE FROM session_anomalies WHERE session_id = ?", (session_id,))
                anom_rows = []
                for a in anomalies:
                    anom_rows.append((
                        session_id,
                        float(a.get("timestamp", start_time)),
                        a.get("anomaly_type", "ANOMALY"),
                        a.get("severity", "INFO"),
                        a.get("description", ""),
                        a.get("recommended_action", ""),
                        float(a.get("metric_value", 0.0)) if a.get("metric_value") is not None else None
                    ))
                cursor.executemany("""
                    INSERT INTO session_anomalies (
                        session_id, timestamp, anomaly_type, severity,
                        description, recommended_action, metric_value
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, anom_rows)

            conn.commit()

        logger.info(f"Persisted session {session_id} to SQLite database ({samples_count} samples).")
        return session_id

    def get_session(self, session_id: str, include_samples: bool = True) -> Optional[StoredSession]:
        """Retrieves a full session with metadata, samples, and anomalies."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, m.subject_id, m.subject_name, m.assessment_type,
                       m.evaluator, m.notes, m.tags, m.updated_at
                FROM sessions s
                LEFT JOIN session_metadata m ON s.session_id = m.session_id
                WHERE s.session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            if not row:
                return None

            row_dict = dict(row)
            tags_raw = row_dict.get("tags")
            tags = json.loads(tags_raw) if tags_raw else []

            metadata = SessionMetadata(
                session_id=session_id,
                subject_id=row_dict.get("subject_id"),
                subject_name=row_dict.get("subject_name"),
                assessment_type=row_dict.get("assessment_type") or AssessmentType.GENERAL_AFFECT.value,
                evaluator=row_dict.get("evaluator"),
                notes=row_dict.get("notes"),
                tags=tags,
                created_at=row_dict.get("updated_at") or row_dict.get("created_at", time.time())
            )

            # Load anomalies
            cursor.execute("""
                SELECT timestamp, anomaly_type, severity, description, recommended_action, metric_value
                FROM session_anomalies
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            anomalies = [dict(r) for r in cursor.fetchall()]

            # Load timeline samples
            timeline = []
            if include_samples:
                cursor.execute("""
                    SELECT timestamp, dominant_emotion, confidence, valence, arousal, dominance,
                           engagement_index, fatigue_level, attention_score, quadrant,
                           probabilities, text_payload
                    FROM session_samples
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """, (session_id,))
                for s_row in cursor.fetchall():
                    probs = json.loads(s_row["probabilities"]) if s_row["probabilities"] else {}
                    txt = json.loads(s_row["text_payload"]) if s_row["text_payload"] else None
                    timeline.append({
                        "timestamp": s_row["timestamp"],
                        "dominant_emotion": s_row["dominant_emotion"],
                        "confidence": s_row["confidence"],
                        "affect": {
                            "valence": s_row["valence"],
                            "arousal": s_row["arousal"],
                            "dominance": s_row["dominance"],
                        },
                        "engagement_index": s_row["engagement_index"],
                        "fatigue_level": s_row["fatigue_level"],
                        "attention_score": s_row["attention_score"],
                        "quadrant": s_row["quadrant"],
                        "probabilities": probs,
                        "text": txt,
                    })

            dist = json.loads(row_dict.get("emotion_distribution") or "{}")
            moments = json.loads(row_dict.get("key_moments") or "[]")

            return StoredSession(
                session_id=session_id,
                start_time=row_dict["start_time"],
                end_time=row_dict["end_time"],
                samples_count=row_dict["samples_count"],
                duration_seconds=row_dict["duration_seconds"],
                average_valence=row_dict["avg_valence"],
                average_arousal=row_dict["avg_arousal"],
                average_dominance=row_dict["avg_dominance"],
                average_engagement=row_dict["avg_engagement"],
                average_fatigue=row_dict["avg_fatigue"],
                average_attention=row_dict["avg_attention"],
                dominant_emotion=row_dict["dominant_emotion"],
                dominant_emotion_distribution=dist,
                key_moments=moments,
                metadata=metadata,
                anomalies=anomalies,
                timeline=timeline,
            )

    def list_sessions(
        self,
        assessment_type: Optional[str] = None,
        tag: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Lists sessions matching optional filters with pagination."""
        query = """
            SELECT s.session_id, s.start_time, s.end_time, s.samples_count, s.duration_seconds,
                   s.avg_valence, s.avg_arousal, s.avg_dominance, s.avg_engagement,
                   s.dominant_emotion, s.created_at,
                   m.subject_id, m.subject_name, m.assessment_type, m.evaluator, m.notes, m.tags
            FROM sessions s
            LEFT JOIN session_metadata m ON s.session_id = m.session_id
            WHERE 1=1
        """
        params: List[Any] = []

        if assessment_type:
            query += " AND m.assessment_type = ?"
            params.append(assessment_type)

        if tag:
            query += " AND m.tags LIKE ?"
            params.append(f"%{tag}%")

        if search_query:
            query += " AND (s.session_id LIKE ? OR m.subject_name LIKE ? OR m.notes LIKE ?)"
            term = f"%{search_query}%"
            params.extend([term, term, term])

        query += " ORDER BY s.start_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                tags_raw = d.get("tags")
                d["tags"] = json.loads(tags_raw) if tags_raw else []
                results.append(d)
            return results

    def update_metadata(self, session_id: str, metadata: Union[SessionMetadata, Dict[str, Any]]) -> bool:
        """Updates metadata for an existing session."""
        if isinstance(metadata, SessionMetadata):
            meta_obj = metadata
        else:
            meta_obj = SessionMetadata.from_dict(metadata)

        tags_json = json.dumps(meta_obj.tags)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_metadata (
                    session_id, subject_id, subject_name, assessment_type,
                    evaluator, notes, tags, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    subject_id=excluded.subject_id,
                    subject_name=excluded.subject_name,
                    assessment_type=excluded.assessment_type,
                    evaluator=excluded.evaluator,
                    notes=excluded.notes,
                    tags=excluded.tags,
                    updated_at=excluded.updated_at
            """, (
                session_id, meta_obj.subject_id, meta_obj.subject_name,
                meta_obj.assessment_type, meta_obj.evaluator, meta_obj.notes,
                tags_json, time.time()
            ))
            conn.commit()
            return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session and cascading samples, metadata, and anomalies."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def migrate_from_json(self, json_dir: Optional[Path] = None) -> int:
        """Migrates legacy flat-file JSON sessions into SQLite database. Idempotent."""
        target_dir = json_dir or SESSIONS_DIR
        if not target_dir.exists():
            return 0

        migrated_count = 0
        json_files = list(target_dir.glob("*.json"))

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session_id = data.get("session_id", jf.stem)
                data["session_id"] = session_id
                self.save_session(data)
                migrated_count += 1
            except Exception as e:
                logger.warning(f"Failed to migrate JSON session {jf.name}: {e}")

        logger.info(f"Migrated {migrated_count} legacy JSON sessions into SQLite.")
        return migrated_count

    def get_stats(self) -> Dict[str, Any]:
        """Calculates global platform repository statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), SUM(samples_count), AVG(duration_seconds) FROM sessions")
            sess_count, total_samples, avg_dur = cursor.fetchone()

            cursor.execute("""
                SELECT m.assessment_type, COUNT(*) 
                FROM session_metadata m 
                GROUP BY m.assessment_type
            """)
            type_counts = {r[0] or "general_affect": r[1] for r in cursor.fetchall()}

            cursor.execute("SELECT COUNT(*) FROM session_anomalies")
            total_anomalies = cursor.fetchone()[0]

            return {
                "total_sessions": sess_count or 0,
                "total_samples": total_samples or 0,
                "average_duration_seconds": round(avg_dur or 0.0, 1),
                "assessment_types": type_counts,
                "total_anomalies_recorded": total_anomalies or 0,
            }
