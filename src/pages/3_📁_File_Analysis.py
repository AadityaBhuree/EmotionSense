"""Page 3: Offline Video, Audio and Dataset File Multimodal Analysis Studio."""

import streamlit as st
import tempfile
import cv2
import soundfile as sf
import numpy as np
from pathlib import Path

from config import THEME_COLORS
from src.ui.styles import inject_modern_styles
from src.ui.components import render_header, render_metric_card, render_affect_summary_badge
from src.ui.charts import render_emotion_radar_chart, render_affect_quadrant_chart, render_emotion_timeline_chart
from src.vision.face_mesh import FaceMeshDetector
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.utils.session_manager import SessionManager

st.set_page_config(page_title="File Analysis Studio | EmotionSense", page_icon="📁", layout="wide")
inject_modern_styles()

render_header("File Multimodal Analysis", "Analyze Pre-Recorded Video, Audio & Speech Transcripts")

st.markdown("""
<div class="es-panel">
    <p style="color: var(--text-sub); font-size: 0.88rem; line-height: 1.5; margin: 0;">
        Ingest Video (MP4, AVI, MOV), Audio (WAV, MP3), or Conversational Datasets (CSV, JSON, TXT) to run offline multimodal emotion extraction and export structured telemetry.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Select Media or Dataset File", type=["mp4", "avi", "mov", "wav", "mp3", "csv", "json", "txt"])

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

        st.markdown("### ⌖ Conversational Telemetry Summary")
        tk1, tk2, tk3, tk4 = st.columns(4)
        with tk1:
            render_metric_card("Total Turns", str(summary.total_turns), delta=f"{len(summary.speakers)} Participants", color="#3b82f6")
        with tk2:
            esc_col = "#10b981" if summary.escalation_risk == "Low" else ("#f59e0b" if summary.escalation_risk == "Moderate" else "#ef4444")
            render_metric_card("Escalation Risk", summary.escalation_risk, delta="Conflict Sentinel", color=esc_col)
        with tk3:
            render_metric_card("Empathy / Rapport", f"{int(summary.rapport_empathy_score * 100)}%", delta="Affective Synchrony", color="#0ea5e9")
        with tk4:
            render_metric_card("Turning Points", str(len(summary.turning_points)), delta="Inflection Shifts", color="#f59e0b")

        st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)
        t_col1, t_col2 = st.columns([5, 6])
        with t_col1:
            st.markdown("<div class='es-section-title'>Message Stream Breakdown</div>", unsafe_allow_html=True)
            for turn in summary.turns[:25]:
                is_r = (turn.speaker == summary.speakers[-1]) if len(summary.speakers) > 1 else False
                render_chat_bubble(turn, is_right=is_r)
            if len(summary.turns) > 25:
                st.caption(f"... and {len(summary.turns) - 25} more messages")

        with t_col2:
            st.markdown("<div class='es-section-title'>Affective Trajectory Timeline</div>", unsafe_allow_html=True)
            st.plotly_chart(render_conversation_flow_chart(summary.emotional_trajectory), use_container_width=True)

            if summary.turning_points:
                st.markdown("<div class='es-section-title'>⚡ Key Emotional Turning Points</div>", unsafe_allow_html=True)
                for tp in summary.turning_points:
                    st.markdown(f"""
                    <div style="background: var(--surface-card); border: 1px solid var(--border-color); border-left: 3px solid #f59e0b; padding: 6px 12px; border-radius: 4px; margin-bottom: 6px; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;">
                        <b>Turn #{tp['turn_index']} [{tp['speaker']}]</b>: <code>{tp['from_emotion']}</code> ➔ <code style="color: #f59e0b;">{tp['to_emotion']}</code> (ΔValence: {tp['valence_delta']:+.2f})
                    </div>
                    """, unsafe_allow_html=True)

    # 2. VIDEO / AUDIO MEDIA FILE ANALYSIS
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        st.markdown("### ⌖ Media Processing Scope")

        if file_ext in [".mp4", ".avi", ".mov"]:
            st.video(tmp_path)
            
            if st.button("🚀 Run Frame-by-Frame Video Affect Extraction", use_container_width=True):
                with st.spinner("Processing video frames via MediaPipe FaceMesh..."):
                    cap = cv2.VideoCapture(tmp_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    detector = FaceMeshDetector()
                    classifier = FacialEmotionClassifier()
                    fusion = MultimodalFusionEngine()
                    
                    samples = []
                    step = max(1, int(fps / 5))  # 5 samples per sec
                    idx = 0
                    
                    prog = st.progress(0)
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if idx % step == 0:
                            v_res = classifier.classify(frame)
                            f_state = fusion.fuse(v_res, None)
                            samples.append(f_state)
                            prog.progress(min(1.0, idx / max(1, frame_count)))
                        idx += 1
                    cap.release()
                    prog.progress(1.0)

                    st.success(f"Completed analysis of {len(samples)} temporal checkpoints.")
                    st.session_state.file_samples = samples

            if "file_samples" in st.session_state and st.session_state.file_samples:
                samples = st.session_state.file_samples
                st.markdown("<div class='es-section-title'>Extracted Video Affect Telemetry</div>", unsafe_allow_html=True)
                st.plotly_chart(render_emotion_timeline_chart(samples), use_container_width=True)

        elif file_ext in [".wav", ".mp3"]:
            st.audio(tmp_path)
            if st.button("🚀 Run Acoustic Prosody & Vocal Emotion Extraction", use_container_width=True):
                with st.spinner("Extracting F0, jitter, shimmer and RMS energy..."):
                    extractor = AcousticProsodyExtractor()
                    voice_clf = VoiceSentimentClassifier()
                    
                    prosody = extractor.extract(tmp_path)
                    v_res = voice_clf.classify_prosody(prosody)
                    
                    st.success("Acoustic analysis complete.")
                    
                    pk1, pk2, pk3, pk4 = st.columns(4)
                    with pk1:
                        render_metric_card("Vocal Emotion", v_res.dominant_emotion.upper(), delta=f"{int(v_res.confidence*100)}% Conf", color="#3b82f6")
                    with pk2:
                        render_metric_card("Mean F0 Pitch", f"{prosody.mean_f0:.1f} Hz", color="#0ea5e9")
                    with pk3:
                        render_metric_card("Pitch Jitter", f"{prosody.jitter:.4f}", color="#f59e0b")
                    with pk4:
                        render_metric_card("Amplitude Shimmer", f"{prosody.shimmer:.4f}", color="#10b981")
                    
                    a1, a2 = st.columns(2)
                    with a1:
                        st.markdown("<div class='es-section-title'>Vocal Emotion Polar Radar</div>", unsafe_allow_html=True)
                        st.plotly_chart(render_emotion_radar_chart(v_res.probabilities), use_container_width=True)
                    with a2:
                        st.markdown("<div class='es-section-title'>Acoustic Affect Coordinates</div>", unsafe_allow_html=True)
                        st.plotly_chart(render_affect_quadrant_chart(v_res.affect), use_container_width=True)
