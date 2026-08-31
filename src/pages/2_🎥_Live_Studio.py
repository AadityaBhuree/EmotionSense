"""Page 1: Real-Time Multimodal Streaming Studio with WebRTC & Telemetry."""

import streamlit as st
import time
import cv2
import numpy as np

from config import THEME_COLORS, EMOTION_COLORS
from src.ui.styles import inject_glassmorphic_styles
from src.ui.components import render_header, render_metric_card, render_affect_summary_badge, render_state_indicators
from src.ui.charts import (
    render_emotion_radar_chart,
    render_affect_quadrant_chart,
    render_emotion_timeline_chart,
    render_action_units_bar_chart,
)
from src.vision.face_mesh import FaceMeshDetector
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.utils.session_manager import SessionManager

# Page Configuration
st.set_page_config(page_title="Live Multimodal Studio | EmotionSense", page_icon="🎥", layout="wide")
inject_glassmorphic_styles()

render_header("Live Multimodal Studio", "Real-Time Face Mesh Tracking, Acoustic Prosody & Affective Telemetry")

# Initialize Session State Objects
if "fusion_engine" not in st.session_state:
    st.session_state.fusion_engine = MultimodalFusionEngine(window_size=30)
if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()
if "face_detector" not in st.session_state:
    st.session_state.face_detector = FaceMeshDetector()
if "emotion_classifier" not in st.session_state:
    st.session_state.emotion_classifier = FacialEmotionClassifier()

# Studio Controls Bar
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
with ctrl_col1:
    stream_source = st.radio("Media Feed Source", ["Live Webcam / Camera", "Synthetic Live Stream Simulator"], horizontal=True)

with ctrl_col2:
    if not st.session_state.session_manager.is_recording:
        if st.button("🔴 Start Recording", use_container_width=True):
            st.session_state.session_manager.start_recording()
            st.rerun()
    else:
        if st.button("⏹️ Stop & Save", use_container_width=True):
            summary = st.session_state.session_manager.stop_recording()
            file_path = st.session_state.session_manager.save_session()
            st.success(f"Session saved to {file_path.name}")
            st.rerun()

with ctrl_col3:
    if st.button("📸 Capture Snapshot", use_container_width=True):
        cur = getattr(st.session_state, "latest_state", None)
        if cur:
            st.session_state.pinned_snapshot = cur.to_dict()
            st.toast("📸 Affect Snapshot Captured!")
        else:
            st.warning("No active stream data to capture.")
    
    recording_status = "RECORDING" if st.session_state.session_manager.is_recording else "IDLE"
    pill_class = "es-pill-active" if st.session_state.session_manager.is_recording else "es-pill-idle"
    st.markdown(f"""
    <div style="padding-top: 5px;">
        <span class="es-pill {pill_class}">● {recording_status}</span>
        <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">{len(st.session_state.session_manager.samples)} samples</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Layout: Left side Video Feed & Actions, Right side Live Telemetry Charts
left_col, right_col = st.columns([5, 6])

# Video & Synthetic Stream Processing
with left_col:
    st.markdown("#### 👁️ Real-Time Visual Stream")

    if stream_source == "Live Webcam / Camera":
        try:
            from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
            from src.ui.video_processor import MultimodalVideoProcessor
            
            webrtc_ctx = webrtc_streamer(
                key="emotion-sense-live",
                video_processor_factory=MultimodalVideoProcessor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            if webrtc_ctx.video_processor:
                latest_state = webrtc_ctx.video_processor.get_latest_state()
                if latest_state:
                    st.session_state.latest_state = latest_state
                    st.session_state.session_manager.add_sample(latest_state)
        except Exception as e:
            st.warning(f"WebRTC not active. Running in simulation mode: {e}")
            stream_source = "Synthetic Live Stream Simulator"

    if stream_source == "Synthetic Live Stream Simulator":
        # Interactive Simulator Controls
        st.markdown('<div class="es-card">', unsafe_allow_html=True)
        st.markdown("**🎮 Real-Time Affect Generator Simulation**")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            sim_emotion = st.selectbox("Target Simulated Emotion", ["joy", "surprise", "sadness", "anger", "neutral", "fear", "disgust"])
        with sim_col2:
            sim_intensity = st.slider("Affect Intensity", 0.1, 1.0, 0.85)

        # Generate synthetic vision & audio result
        probs = {emo: 0.05 for emo in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "contempt"]}
        probs[sim_emotion] = float(sim_intensity)
        # normalize
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        from src.core.types import VisionEmotionResult, VoiceEmotionResult, FacialActionUnits, AcousticFeatures
        synthetic_aus = FacialActionUnits(
            lip_corner_puller=0.8 if sim_emotion == "joy" else 0.1,
            brow_lowerer=0.7 if sim_emotion == "anger" else 0.05,
            jaw_drop=0.75 if sim_emotion in ["surprise", "fear"] else 0.1,
            lip_corner_depressor=0.6 if sim_emotion == "sadness" else 0.05,
        )
        synthetic_vision = VisionEmotionResult(
            face_detected=True,
            dominant_emotion=sim_emotion,
            confidence=sim_intensity,
            probabilities=probs,
            action_units=synthetic_aus,
            head_pose={"yaw": np.random.uniform(-5, 5), "pitch": np.random.uniform(-3, 3), "roll": 0.0}
        )
        synthetic_state = st.session_state.fusion_engine.fuse(synthetic_vision, None, text=live_text_res)
        st.session_state.latest_state = synthetic_state
        st.session_state.session_manager.add_sample(synthetic_state)

        # Render Synthetic Frame Canvas
        canvas = np.zeros((320, 480, 3), dtype=np.uint8)
        canvas[:] = (15, 23, 42)
        # Draw face oval
        cv2.ellipse(canvas, (240, 160), (90, 120), 0, 0, 360, (99, 102, 241), 2)
        # Draw eyes
        cv2.circle(canvas, (205, 130), 8, (6, 182, 212), -1)
        cv2.circle(canvas, (275, 130), 8, (6, 182, 212), -1)
        # Draw mouth based on emotion
        if sim_emotion == "joy":
            cv2.ellipse(canvas, (240, 200), (40, 20), 0, 0, 180, (16, 185, 129), 3)
        elif sim_emotion == "sadness":
            cv2.ellipse(canvas, (240, 220), (35, 15), 0, 180, 360, (59, 130, 246), 3)
        else:
            cv2.line(canvas, (215, 205), (265, 205), (148, 163, 184), 2)

        cv2.putText(canvas, f"SIMULATED: {sim_emotion.upper()} ({int(sim_intensity*100)}%)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (248, 250, 252), 1, cv2.LINE_AA)

        st.image(canvas, channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Live Text & Spoken Prompt Input for True Tri-Modal Fusion
    st.markdown("#### 💬 Live Text & Spoken Prompt (Tri-Modal Fusion)")
    live_prompt = st.text_input(
        "Spoken / Typed Phrase Context",
        placeholder="Type or paste what the person is saying (e.g., 'I am so excited for this!' or 'I'm furious')...",
        key="live_stream_text_prompt"
    )
    live_text_res = None
    if live_prompt.strip():
        if "text_classifier" not in st.session_state:
            from src.text import TextEmotionClassifier
            st.session_state.text_classifier = TextEmotionClassifier()
        live_text_res = st.session_state.text_classifier.analyze_text(live_prompt)
        st.markdown(f"""
        <div style="font-size: 0.8rem; background: rgba(30, 41, 59, 0.4); padding: 4px 10px; border-radius: 6px; margin-bottom: 8px;">
            <b>NLP Modality Detected:</b> <span style="color: #818cf8;">{live_text_res.dominant_emotion.upper()}</span> ({int(live_text_res.confidence*100)}%)
        </div>
        """, unsafe_allow_html=True)

    # Action Units Section
    current_state = getattr(st.session_state, "latest_state", None)
    if current_state and current_state.vision:
        st.markdown("#### 🧬 Facial Action Units (FACS)")
        st.plotly_chart(render_action_units_bar_chart(current_state.vision.action_units), use_container_width=True)

# Telemetry Charts & Affect Dimension
with right_col:
    st.markdown("#### ⚡ Real-Time Affect Telemetry")

    current_state = getattr(st.session_state, "latest_state", None)
    if current_state:
        render_affect_summary_badge(current_state)
        render_state_indicators(current_state)

        tab1, tab2 = st.tabs(["📊 Radar & Circumplex", "📈 Dynamic Timeline"])
        with tab1:
            t1, t2 = st.columns(2)
            with t1:
                st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8;'>8-Emotion Polar Radar</p>", unsafe_allow_html=True)
                st.plotly_chart(render_emotion_radar_chart(current_state.probabilities), use_container_width=True)
            with t2:
                st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8;'>2D Affect Quadrant (V-A)</p>", unsafe_allow_html=True)
                st.plotly_chart(render_affect_quadrant_chart(current_state.affect), use_container_width=True)

        with tab2:
            st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8;'>Real-Time Affect & Behavioral Stream</p>", unsafe_allow_html=True)
            recent_history = st.session_state.fusion_engine.get_recent_history()
            st.plotly_chart(render_emotion_timeline_chart(recent_history), use_container_width=True)
    else:
        st.info("Start video stream or simulator above to generate live multimodal telemetry.")

if "pinned_snapshot" in st.session_state:
    st.markdown("---")
    st.markdown("### 📸 Pinned Affect Snapshot")
    snap = st.session_state.pinned_snapshot
    sn1, sn2, sn3, sn4 = st.columns(4)
    with sn1:
        render_metric_card("Snapshot Affect", str(snap.get("dominant_emotion", "neutral")).upper(), delta=f"{int(snap.get('confidence', 0)*100)}% Conf", color="#6366f1")
    with sn2:
        render_metric_card("Snapshot Valence", f"{snap.get('affect', {}).get('valence', 0):+.2f}", color="#10b981")
    with sn3:
        render_metric_card("Snapshot Arousal", f"{snap.get('affect', {}).get('arousal', 0):+.2f}", color="#f59e0b")
    with sn4:
        import json
        snap_json = json.dumps(snap, indent=2)
        st.download_button("📥 Download Snapshot JSON", data=snap_json, file_name="affect_snapshot.json", mime="application/json", use_container_width=True)

