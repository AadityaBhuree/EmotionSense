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
        Upload an MP4, AVI, WAV, or MP3 file to run comprehensive multimodal affect extraction, generate timeline timelines, and export detailed affective diagnostic reports.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a Video or Audio File", type=["mp4", "avi", "mov", "wav", "mp3"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_ext = Path(uploaded_file.name).suffix.lower()
    
    st.success(f"Loaded `{uploaded_file.name}` ({len(file_bytes) / 1024 / 1024:.2f} MB)")

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
        dist_cols = st.columns(len(summary.dominant_emotion_distribution))
        for i, (emo, pct) in enumerate(summary.dominant_emotion_distribution.items()):
            with dist_cols[i]:
                st.metric(emo.capitalize(), f"{int(pct * 100)}%")
