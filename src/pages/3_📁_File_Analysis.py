"""Page 2: Offline Video and Audio File Multimodal Analysis Studio."""

import streamlit as st
import tempfile
import cv2
import soundfile as sf
import numpy as np
from pathlib import Path

from config import THEME_COLORS
from src.ui.styles import inject_glassmorphic_styles
from src.ui.components import render_header, render_metric_card, render_affect_summary_badge
from src.ui.charts import render_emotion_radar_chart, render_affect_quadrant_chart, render_emotion_timeline_chart
from src.vision.face_mesh import FaceMeshDetector
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.utils.session_manager import SessionManager

st.set_page_config(page_title="File Analysis Studio | EmotionSense", page_icon="📁", layout="wide")
inject_glassmorphic_styles()

render_header("File Multimodal Analysis", "Analyze Pre-Recorded Video, Audio & Speech Recordings")

st.markdown("""
<div class="es-card">
    <p style="color: #94a3b8; margin-bottom: 0;">
        Upload a Video (MP4, AVI, MOV), Audio (WAV, MP3), or Conversational Text Dataset (CSV, JSON, TXT) to run comprehensive affective intelligence extraction and export detailed diagnostic reports.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a File (Video, Audio, or Text / Chat Transcript)", type=["mp4", "avi", "mov", "wav", "mp3", "csv", "json", "txt"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_ext = Path(uploaded_file.name).suffix.lower()
    
    st.success(f"Loaded `{uploaded_file.name}` ({len(file_bytes) / 1024:.1f} KB)")

    # 1. TEXT / CHAT TRANSCRIPT FILE ANALYSIS
    if file_ext in [".csv", ".json", ".txt"]:
        from src.text import ConversationAffectAnalyzer, TextEmotionClassifier
        from src.ui.charts import render_conversation_flow_chart, render_emotion_distribution_pie
        from src.ui.components import render_chat_bubble

        analyzer = ConversationAffectAnalyzer()
        raw_content = file_bytes.decode("utf-8", errors="ignore")

        if file_ext == ".csv":
            import pandas as pd
            import io
            df_in = pd.read_csv(io.StringIO(raw_content))
            str_cols = [c for c in df_in.columns if df_in[c].dtype == "object"]
            if str_cols:
                lines = df_in[str_cols[0]].dropna().astype(str).tolist()
            else:
                lines = df_in.iloc[:, 0].dropna().astype(str).tolist()
            summary = analyzer.parse_and_analyze_transcript("\n".join(lines))
        else:
            summary = analyzer.parse_and_analyze_transcript(raw_content)

        st.markdown("### 📊 Conversational Telemetry Summary")
        tk1, tk2, tk3, tk4 = st.columns(4)
        with tk1:
            render_metric_card("Total Turns", str(summary.total_turns), delta=f"{len(summary.speakers)} Participants", color="#6366f1")
        with tk2:
            esc_col = "#10b981" if summary.escalation_risk == "Low" else ("#f59e0b" if summary.escalation_risk == "Moderate" else "#ef4444")
            render_metric_card("Escalation Risk", summary.escalation_risk, delta="Conflict Watchdog", color=esc_col)
        with tk3:
            render_metric_card("Empathy / Rapport", f"{int(summary.rapport_empathy_score * 100)}%", delta="Affective Synchrony", color="#06b6d4")
        with tk4:
            render_metric_card("Turning Points", str(len(summary.turning_points)), delta="Inflection Shifts", color="#f59e0b")

        st.markdown("<br>", unsafe_allow_html=True)
        t_col1, t_col2 = st.columns([5, 6])
        with t_col1:
            st.markdown("#### 💬 Message Stream Breakdown")
            for turn in summary.turns[:25]:
                is_r = (turn.speaker == summary.speakers[-1]) if len(summary.speakers) > 1 else False
                render_chat_bubble(turn, is_right=is_r)
            if len(summary.turns) > 25:
                st.caption(f"... and {len(summary.turns) - 25} more messages")

        with t_col2:
            st.markdown("#### 📈 Affective Trajectory Timeline")
            st.plotly_chart(render_conversation_flow_chart(summary.emotional_trajectory), use_container_width=True)

            if summary.turning_points:
                st.markdown("##### ⚡ Key Emotional Turning Points")
                for tp in summary.turning_points:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.45); border-left: 3px solid #f59e0b; padding: 6px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 0.85rem;">
                        <b>Turn #{tp['turn_index']} ({tp['speaker']})</b>: <code>{tp['from_emotion']}</code> ➔ <code style="color: #f59e0b;">{tp['to_emotion']}</code> (ΔValence: {tp['valence_delta']:+.2f})
                    </div>
                    """, unsafe_allow_html=True)

    # 2. VIDEO / AUDIO MEDIA FILE ANALYSIS
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            if file_ext in [".mp4", ".avi", ".mov"]:
                st.video(file_bytes)
            elif file_ext in [".wav", ".mp3"]:
                st.audio(file_bytes)

        with col2:
            if st.button("🚀 Run Full Multimodal Analysis", use_container_width=True):
                with st.spinner("Processing media frames & extracting acoustic features..."):
                    face_mesh = FaceMeshDetector()
                    emotion_classifier = FacialEmotionClassifier()
                    prosody_extractor = AcousticProsodyExtractor()
                    voice_classifier = VoiceSentimentClassifier()
                    fusion_engine = MultimodalFusionEngine(window_size=20)
                    session_mgr = SessionManager(session_id=f"file_{Path(uploaded_file.name).stem}")
                    session_mgr.start_recording()

                    # Process with temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name

                    if file_ext in [".mp4", ".avi", ".mov"]:
                        cap = cv2.VideoCapture(tmp_path)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
                        sample_stride = max(fps // 5, 1)  # 5 samples per sec

                        idx = 0
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if idx % sample_stride == 0:
                                landmarks, head_pose = face_mesh.process_frame(frame)
                                vis_res = emotion_classifier.classify_emotion(landmarks, head_pose)
                                fused = fusion_engine.fuse(vis_res, None)
                                session_mgr.add_sample(fused)
                            idx += 1
                        cap.release()

                    summary = session_mgr.stop_recording()
                    st.session_state.file_summary = summary
                    st.session_state.file_samples = session_mgr.samples
                    st.success("Analysis Complete!")

        # Display Analysis Results
        if "file_summary" in st.session_state:
            summary = st.session_state.file_summary
            samples = st.session_state.file_samples

            st.markdown("---")
            st.markdown("### 📊 Diagnostic Results Summary")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Total Samples", f"{summary.samples_count}", color="#6366f1")
            with m2:
                render_metric_card("Avg Engagement", f"{int(summary.average_engagement * 100)}%", color="#06b6d4")
            with m3:
                render_metric_card("Avg Valence", f"{summary.average_affect['valence']:+.2f}", color="#10b981")
            with m4:
                render_metric_card("Avg Arousal", f"{summary.average_affect['arousal']:+.2f}", color="#f59e0b")

            st.markdown("#### 📈 Multi-Dimensional Timeline Stream")
            st.plotly_chart(render_emotion_timeline_chart(samples), use_container_width=True)

            st.markdown("#### 🥧 Dominant Emotion Distribution")
            dist_cols = st.columns(max(1, len(summary.dominant_emotion_distribution)))
            for i, (emo, pct) in enumerate(summary.dominant_emotion_distribution.items()):
                with dist_cols[i]:
                    st.metric(emo.capitalize(), f"{int(pct * 100)}%")

            # Executive Report Download
            st.markdown("---")
            st.markdown("### 📑 Executive Diagnostic Report")
            report_md = f"""# EmotionSense Media Diagnostic Report
- **File Name:** {uploaded_file.name}
- **Samples Processed:** {summary.samples_count}
- **Average Engagement:** {int(summary.average_engagement * 100)}%
- **Average Valence (Mood):** {summary.average_affect['valence']:+.2f}
- **Average Arousal (Energy):** {summary.average_affect['arousal']:+.2f}
- **Dominant Emotion Distribution:**
{chr(10).join([f'  - **{k.capitalize()}:** {int(v*100)}%' for k, v in summary.dominant_emotion_distribution.items()])}
"""
            st.download_button(
                "📥 Download Diagnostic Summary Report (Markdown)",
                data=report_md,
                file_name=f"{Path(uploaded_file.name).stem}_diagnostic_report.md",
                mime="text/markdown",
                use_container_width=True
            )


