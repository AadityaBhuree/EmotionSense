"""Page 4: Session History, Timeline Scrubbing & Diagnostic Report Exporter."""

import streamlit as st
import pandas as pd
import json

from config import THEME_COLORS
from src.ui.styles import inject_modern_styles
from src.ui.components import render_header, render_metric_card, render_affect_summary_badge
from src.ui.charts import render_emotion_radar_chart, render_affect_quadrant_chart, render_emotion_timeline_chart
from src.utils.session_manager import SessionManager
from src.utils.report_generator import DiagnosticReportGenerator
from src.fusion.anomaly_detector import AffectiveAnomalyDetector
from src.core.types import MultimodalEmotionState, AffectVector, SessionRecord

st.set_page_config(page_title="Session Intelligence & History | EmotionSense", page_icon="📑", layout="wide")
inject_modern_styles()

render_header("Session Intelligence & History", "Review Historical Affective Logs, Scrub Timelines & Export Data")

saved_files = SessionManager.list_saved_sessions()

if not saved_files:
    st.info("No saved sessions found in `data/sessions/`. Start a live recording in the **Live Studio** to generate session records.")
else:
    col_sel, col_del = st.columns([4, 1])
    with col_sel:
        selected_file = st.selectbox(
            "Select Session Record",
            saved_files,
            format_func=lambda x: f"📁 {x.stem} ({x.stat().st_size / 1024:.1f} KB)"
        )
    with col_del:
        if st.button("🗑️ Delete Session", use_container_width=True):
            selected_file.unlink()
            st.success("Deleted session.")
            st.rerun()

    session_data = SessionManager.load_session(selected_file)

    st.markdown("<div class='es-section-title'>📊 Session Aggregate Telemetry</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Samples", f"{session_data.get('samples_count', 0)}", color="#3b82f6")
    with m2:
        render_metric_card("Avg Engagement", f"{int(session_data.get('average_engagement', 0) * 100)}%", color="#0ea5e9")
    with m3:
        render_metric_card("Avg Valence", f"{session_data.get('average_affect', {}).get('valence', 0):+.2f}", color="#10b981")
    with m4:
        render_metric_card("Avg Arousal", f"{session_data.get('average_affect', {}).get('arousal', 0):+.2f}", color="#f59e0b")

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

    if detected_events:
        st.markdown(f"<div class='es-section-title'>🛡️ Affective Anomaly & Escalation Sentinel ({len(detected_events)} Events)</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(detected_events), use_container_width=True)

    # Export Section
    st.markdown("<div class='es-section-title'>💾 Export Telemetry Dataset & Clinical Reports</div>", unsafe_allow_html=True)

    session_rec = SessionRecord(
        session_id=session_data.get("session_id", selected_file.stem),
        start_time=session_data.get("start_time", 0.0),
        end_time=session_data.get("end_time", 0.0),
        samples_count=session_data.get("samples_count", len(timeline_points)),
        timeline=timeline_points,
        average_affect=session_data.get("average_affect", {}),
        dominant_emotion_distribution=session_data.get("dominant_emotion_distribution", {}),
        average_engagement=session_data.get("average_engagement", 0.0),
        average_fatigue=session_data.get("average_fatigue", 0.0),
        average_attention=session_data.get("average_attention", 0.0),
        key_moments=key_moments,
    )

    html_report_str = DiagnosticReportGenerator.generate_html_report(session_rec, detected_events)
    md_report_str = DiagnosticReportGenerator.generate_markdown_report(session_rec, detected_events)

    exp1, exp2, exp3, exp4 = st.columns(4)
    with exp1:
        st.download_button(
            "📄 Diagnostic Report (HTML)",
            data=html_report_str,
            file_name=f"{selected_file.stem}_diagnostic_report.html",
            mime="text/html",
            use_container_width=True
        )
    with exp2:
        st.download_button(
            "📝 Clinical Summary (MD)",
            data=md_report_str,
            file_name=f"{selected_file.stem}_clinical_summary.md",
            mime="text/markdown",
            use_container_width=True
        )
    with exp3:
        json_str = json.dumps(session_data, indent=2)
        st.download_button(
            "📥 Raw Session (JSON)",
            data=json_str,
            file_name=f"{selected_file.stem}.json",
            mime="application/json",
            use_container_width=True
        )
    with exp4:
        if timeline_points:
            df = pd.DataFrame(timeline_points)
            csv_str = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Timeline (CSV)",
                data=csv_str,
                file_name=f"{selected_file.stem}_timeline.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Multi-Session Comparative Benchmarking
    if len(saved_files) > 1:
        st.markdown("<div class='es-section-title'>📈 Multi-Session Comparative Benchmarking</div>", unsafe_allow_html=True)
        benchmarks = []
        for sf_path in saved_files:
            s_data = SessionManager.load_session(sf_path)
            benchmarks.append({
                "Session": sf_path.stem,
                "Samples": s_data.get("samples_count", 0),
                "Avg Valence": round(s_data.get("average_affect", {}).get("valence", 0.0), 2),
                "Avg Arousal": round(s_data.get("average_affect", {}).get("arousal", 0.0), 2),
                "Avg Engagement": f"{int(s_data.get('average_engagement', 0) * 100)}%",
            })
        st.dataframe(pd.DataFrame(benchmarks), use_container_width=True)
