"""Page 4: Session History, Timeline Scrubbing, SQLite Persistence & Clinical PDF Exporter."""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from config import THEME_COLORS, SESSIONS_DIR
from src.ui.styles import inject_modern_styles
from src.ui.components import render_header, render_metric_card
from src.ui.charts import render_emotion_radar_chart, render_affect_quadrant_chart
from src.utils.session_manager import SessionManager
from src.utils.report_generator import DiagnosticReportGenerator
from src.utils.pdf_exporter import ClinicalPDFExporter
from src.storage import AssessmentType, SessionMetadata
from src.fusion.anomaly_detector import AffectiveAnomalyDetector
from src.core.types import MultimodalEmotionState, AffectVector, SessionRecord

st.set_page_config(page_title="Session Intelligence & History | EmotionSense", page_icon="📑", layout="wide")
inject_modern_styles()

render_header("Session Intelligence & Clinical Records", "Review Longitudinal Affect Logs, Manage Metadata & Export Clinical Diagnostics")

db = SessionManager.get_database()

# Toolbar for search and sync
tb_col1, tb_col2, tb_col3 = st.columns([2, 1.5, 1])
with tb_col1:
    search_term = st.text_input("🔍 Search Sessions", placeholder="Search by Session ID, Subject, or Notes...")
with tb_col2:
    assessment_filter = st.selectbox(
        "Filter by Assessment Type",
        ["All"] + list(AssessmentType.display_names().keys()),
        format_func=lambda x: "All Assessment Types" if x == "All" else AssessmentType.display_names().get(x, x)
    )
with tb_col3:
    st.write("")
    st.write("")
    if st.button("🔄 Sync JSON to DB", use_container_width=True, help="Scan and import legacy flat-file JSON sessions into SQLite"):
        count = db.migrate_from_json()
        st.success(f"Synchronized {count} sessions.")
        st.rerun()

filter_type = None if assessment_filter == "All" else assessment_filter
db_sessions = db.list_sessions(assessment_type=filter_type, search_query=search_term if search_term else None)

# Fallback: if database has zero sessions, trigger one auto-migration from disk
if not db_sessions:
    json_count = len(list(SESSIONS_DIR.glob("*.json")))
    if json_count > 0:
        db.migrate_from_json()
        db_sessions = db.list_sessions(assessment_type=filter_type, search_query=search_term if search_term else None)

if not db_sessions:
    st.info("No recorded sessions found matching criteria. Record live sessions in the **Live Studio** or upload files in **File Analysis** to generate records.")
else:
    col_sel, col_del = st.columns([4, 1])
    with col_sel:
        session_options = [s["session_id"] for s in db_sessions]
        session_labels = {}
        for s in db_sessions:
            s_name = s.get("subject_name") or "Anonymous"
            s_type = AssessmentType.display_names().get(s.get("assessment_type", ""), "General")
            dt_label = datetime.fromtimestamp(s.get("start_time", 0)).strftime("%Y-%m-%d %H:%M")
            session_labels[s["session_id"]] = f"📁 {s['session_id']} | {s_name} ({s_type}) | {dt_label} [{s.get('samples_count', 0)} frames]"

        selected_id = st.selectbox(
            "Select Session Record",
            session_options,
            format_func=lambda x: session_labels.get(x, x)
        )
    with col_del:
        st.write("")
        st.write("")
        if st.button("🗑️ Delete Session", use_container_width=True):
            db.delete_session(selected_id)
            json_file = SESSIONS_DIR / f"{selected_id}.json"
            if json_file.exists():
                json_file.unlink()
            st.success(f"Deleted session '{selected_id}'.")
            st.rerun()

    stored_session = db.get_session(selected_id, include_samples=True)
    if not stored_session:
        st.error("Could not load session details.")
        st.stop()

    session_data = stored_session.to_dict()
    metadata_obj = stored_session.metadata

    # Metadata Card & Editor
    with st.expander("👤 Subject & Clinical Assessment Metadata (Click to Edit)", expanded=False):
        with st.form("meta_form"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                in_name = st.text_input("Subject / Candidate Name", value=metadata_obj.subject_name or "")
                in_id = st.text_input("Subject / Patient ID", value=metadata_obj.subject_id or "")
            with f_col2:
                in_type = st.selectbox(
                    "Assessment Classification",
                    list(AssessmentType.display_names().keys()),
                    index=list(AssessmentType.display_names().keys()).index(metadata_obj.assessment_type) if metadata_obj.assessment_type in AssessmentType.display_names() else 0,
                    format_func=lambda x: AssessmentType.display_names().get(x, x)
                )
                in_eval = st.text_input("Evaluator / Clinician", value=metadata_obj.evaluator or "")
            with f_col3:
                tags_default = ", ".join(metadata_obj.tags) if metadata_obj.tags else ""
                in_tags = st.text_input("Tags (comma separated)", value=tags_default)
            
            in_notes = st.text_area("Clinical Observations & Diagnostic Notes", value=metadata_obj.notes or "", rows=2)
            
            if st.form_submit_button("💾 Save Metadata Updates"):
                parsed_tags = [t.strip() for t in in_tags.split(",") if t.strip()]
                new_meta = SessionMetadata(
                    session_id=selected_id,
                    subject_id=in_id.strip() or None,
                    subject_name=in_name.strip() or None,
                    assessment_type=in_type,
                    evaluator=in_eval.strip() or None,
                    notes=in_notes.strip() or None,
                    tags=parsed_tags,
                )
                db.update_metadata(selected_id, new_meta)
                st.success("Session metadata updated successfully!")
                st.rerun()

    # Telemetry HUD
    st.markdown("<div class='es-section-title'>📊 Session Aggregate Telemetry</div>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card("Total Samples", f"{session_data.get('samples_count', 0)}", color="#3b82f6")
    with m2:
        render_metric_card("Duration", f"{session_data.get('duration_seconds', 0):.1f}s", color="#8b5cf6")
    with m3:
        render_metric_card("Avg Engagement", f"{int(session_data.get('average_engagement', 0) * 100)}%", color="#0ea5e9")
    with m4:
        render_metric_card("Avg Valence", f"{session_data.get('average_valence', 0):+.2f}", color="#10b981")
    with m5:
        render_metric_card("Avg Arousal", f"{session_data.get('average_arousal', 0):+.2f}", color="#f59e0b")

    timeline_points = session_data.get("timeline", [])

    if timeline_points:
        st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='es-section-title'>⏱️ Interactive Timeline Scrubber</div>", unsafe_allow_html=True)
        
        frame_idx = st.slider("Scrub Timeline Frame", 0, len(timeline_points) - 1, 0)
        current_frame = timeline_points[frame_idx]

        sc1, sc2, sc3 = st.columns([1, 1, 1])
        with sc1:
            st.markdown(f"""
            <div class="es-panel">
                <div class="es-section-title">Frame #{frame_idx} Telemetry</div>
                <div style="font-size: 0.85rem; line-height: 1.6; font-family: 'JetBrains Mono', monospace;">
                    <div>Dominant: <b style="color: #3b82f6;">{current_frame.get('dominant_emotion', 'neutral').upper()}</b></div>
                    <div>Confidence: <b>{int(current_frame.get('confidence', 0)*100)}%</b></div>
                    <div>Quadrant: <b>{current_frame.get('quadrant', 'N/A')}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if current_frame.get("text"):
                txt_obj = current_frame.get("text")
                msg_str = txt_obj.get("text") if isinstance(txt_obj, dict) else str(txt_obj)
                st.caption(f"Spoken text: \"{msg_str}\"")
        with sc2:
            st.plotly_chart(render_emotion_radar_chart(current_frame.get("probabilities", {})), use_container_width=True)
        with sc3:
            affect_dict = current_frame.get("affect", {})
            affect_obj = AffectVector(
                valence=affect_dict.get("valence", 0.0),
                arousal=affect_dict.get("arousal", 0.0),
                dominance=affect_dict.get("dominance", 0.0)
            )
            st.plotly_chart(render_affect_quadrant_chart(affect_obj), use_container_width=True)

    # Key Affective Moments
    key_moments = session_data.get("key_moments", [])
    if key_moments:
        st.markdown("<div class='es-section-title'>⚡ Key Affective Pivot Moments (High Intensity Shifts)</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(key_moments), use_container_width=True)

    # Affective Anomaly Sentinel Analysis
    detector = AffectiveAnomalyDetector()
    for frame in timeline_points:
        aff = frame.get("affect", {})
        st_obj = MultimodalEmotionState(
            timestamp=frame.get("timestamp", 0.0),
            dominant_emotion=frame.get("dominant_emotion", "neutral"),
            confidence=frame.get("confidence", 0.0),
            affect=AffectVector(
                valence=aff.get("valence", 0.0),
                arousal=aff.get("arousal", 0.0),
                dominance=aff.get("dominance", 0.0)
            ),
            engagement_index=frame.get("engagement_index", 0.0),
            fatigue_level=frame.get("fatigue_level", 0.0),
            attention_score=frame.get("attention_score", 0.0),
        )
        detector.process_state(st_obj)

    anom_summary = detector.get_anomaly_summary()
    detected_events = anom_summary.get("events", [])
    if not detected_events and session_data.get("anomalies"):
        detected_events = session_data.get("anomalies", [])

    if detected_events:
        st.markdown(f"<div class='es-section-title'>🛡️ Affective Anomaly & Escalation Sentinel ({len(detected_events)} Events)</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(detected_events), use_container_width=True)

    # Export Section
    st.markdown("<div class='es-section-title'>💾 Export Telemetry Dataset & Clinical Reports</div>", unsafe_allow_html=True)

    session_rec = SessionRecord(
        session_id=stored_session.session_id,
        start_time=stored_session.start_time,
        end_time=stored_session.end_time,
        samples_count=stored_session.samples_count,
        timeline=timeline_points,
        average_affect={
            "valence": stored_session.average_valence,
            "arousal": stored_session.average_arousal,
            "dominance": stored_session.average_dominance,
        },
        dominant_emotion_distribution=stored_session.dominant_emotion_distribution,
        average_engagement=stored_session.average_engagement,
        average_fatigue=stored_session.average_fatigue,
        average_attention=stored_session.average_attention,
        key_moments=key_moments,
    )

    html_report_str = DiagnosticReportGenerator.generate_html_report(session_rec, detected_events)
    md_report_str = DiagnosticReportGenerator.generate_markdown_report(session_rec, detected_events)

    # Generate Clinical PDF
    pdf_bytes = None
    if ClinicalPDFExporter.is_available():
        try:
            pdf_bytes = ClinicalPDFExporter.generate_pdf(
                stored_session,
                anomalies=detected_events,
                metadata=metadata_obj
            )
        except Exception as e:
            st.warning(f"Could not compile PDF: {e}")

    exp_cols = st.columns(5)
    with exp_cols[0]:
        if pdf_bytes:
            st.download_button(
                "🏥 Clinical PDF Report",
                data=pdf_bytes,
                file_name=f"{selected_id}_clinical_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.button("🏥 PDF (Unavailable)", disabled=True, use_container_width=True)
    with exp_cols[1]:
        st.download_button(
            "📄 Diagnostic HTML",
            data=html_report_str,
            file_name=f"{selected_id}_diagnostic_report.html",
            mime="text/html",
            use_container_width=True
        )
    with exp_cols[2]:
        st.download_button(
            "📝 Clinical MD",
            data=md_report_str,
            file_name=f"{selected_id}_clinical_summary.md",
            mime="text/markdown",
            use_container_width=True
        )
    with exp_cols[3]:
        json_str = json.dumps(session_data, indent=2)
        st.download_button(
            "📥 Session JSON",
            data=json_str,
            file_name=f"{selected_id}.json",
            mime="application/json",
            use_container_width=True
        )
    with exp_cols[4]:
        if timeline_points:
            df = pd.DataFrame(timeline_points)
            csv_str = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Timeline CSV",
                data=csv_str,
                file_name=f"{selected_id}_timeline.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Multi-Session Comparative Benchmarking
    if len(db_sessions) > 1:
        st.markdown("<div class='es-section-title'>📈 Multi-Session Comparative Benchmarking</div>", unsafe_allow_html=True)
        benchmarks = []
        for s_row in db_sessions:
            benchmarks.append({
                "Session ID": s_row["session_id"],
                "Subject": s_row.get("subject_name") or "Anonymous",
                "Assessment Type": AssessmentType.display_names().get(s_row.get("assessment_type", ""), "General"),
                "Samples": s_row.get("samples_count", 0),
                "Avg Valence": round(s_row.get("avg_valence", 0.0), 2),
                "Avg Arousal": round(s_row.get("avg_arousal", 0.0), 2),
                "Avg Engagement": f"{int(s_row.get('avg_engagement', 0.0) * 100)}%",
            })
        st.dataframe(pd.DataFrame(benchmarks), use_container_width=True)
