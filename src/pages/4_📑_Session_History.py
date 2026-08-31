"""Page 3: Session History, Timeline Scrubbing & Diagnostic Report Exporter."""

import streamlit as st
import pandas as pd
import json

from config import THEME_COLORS
from src.ui.styles import inject_glassmorphic_styles
from src.ui.components import render_header, render_metric_card, render_affect_summary_badge
from src.ui.charts import render_emotion_radar_chart, render_affect_quadrant_chart, render_emotion_timeline_chart
from src.utils.session_manager import SessionManager
from src.core.types import MultimodalEmotionState, AffectVector

st.set_page_config(page_title="Session Intelligence & History | EmotionSense", page_icon="📑", layout="wide")
inject_glassmorphic_styles()

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

    st.markdown("---")
    st.markdown("### 📊 Session Aggregate Summary")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Samples", f"{session_data.get('samples_count', 0)}", color="#6366f1")
    with m2:
        render_metric_card("Avg Engagement", f"{int(session_data.get('average_engagement', 0) * 100)}%", color="#06b6d4")
    with m3:
        render_metric_card("Avg Valence", f"{session_data.get('average_affect', {}).get('valence', 0):+.2f}", color="#10b981")
    with m4:
        render_metric_card("Avg Arousal", f"{session_data.get('average_affect', {}).get('arousal', 0):+.2f}", color="#f59e0b")

    timeline_points = session_data.get("timeline", [])

    if timeline_points:
        st.markdown("---")
        st.markdown("### ⏱️ Interactive Timeline Scrubber")
        
        frame_idx = st.slider("Scrub Timeline Frame", 0, len(timeline_points) - 1, 0)
        current_frame = timeline_points[frame_idx]

        sc1, sc2, sc3 = st.columns([1, 1, 1])
        with sc1:
            st.markdown(f"**Dominant Emotion:** `{current_frame.get('dominant_emotion', 'neutral').upper()}`")
            st.markdown(f"**Confidence:** `{int(current_frame.get('confidence', 0)*100)}%`")
            st.markdown(f"**Affect Quadrant:** `{current_frame.get('quadrant', 'N/A')}`")
            if current_frame.get("text"):
                txt_obj = current_frame.get("text")
                msg_str = txt_obj.get("text") if isinstance(txt_obj, dict) else str(txt_obj)
                st.markdown(f"**Message:** *\"{msg_str}\"*")
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
        st.markdown("---")
        st.markdown("### ⚡ Key Affective Pivot Moments (High Intensity Shifts)")
        st.dataframe(pd.DataFrame(key_moments), use_container_width=True)

    # Export Section
    st.markdown("---")
    st.markdown("### 💾 Export Telemetry Dataset")
    exp1, exp2 = st.columns(2)
    with exp1:
        json_str = json.dumps(session_data, indent=2)
        st.download_button(
            "📥 Download Raw Session (JSON)",
            data=json_str,
            file_name=f"{selected_file.stem}.json",
            mime="application/json",
            use_container_width=True
        )
    with exp2:
        if timeline_points:
            df = pd.DataFrame(timeline_points)
            csv_str = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Telemetry Timeline (CSV)",
                data=csv_str,
                file_name=f"{selected_file.stem}_timeline.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Multi-Session Comparative Benchmarking
    if len(saved_files) > 1:
        st.markdown("---")
        st.markdown("### 📈 Multi-Session Comparative Benchmarking")
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

