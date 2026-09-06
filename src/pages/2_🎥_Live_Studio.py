"""Page 2: Real-Time Multimodal Streaming Studio with WebRTC & Telemetry."""

import streamlit as st
import time
import cv2
import numpy as np

from config import THEME_COLORS, EMOTION_COLORS
from src.ui.styles import inject_modern_styles
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
from src.ui.video_processor import (
    MultimodalStreamContext,
    MultimodalAudioProcessor,
    MultimodalVideoProcessor,
)

# Page Configuration
st.set_page_config(page_title="Live Multimodal Studio | EmotionSense", page_icon="🎥", layout="wide")
inject_modern_styles()

render_header("Live Multimodal Studio", "Real-Time 3D Face Mesh, Acoustic Prosody & Affect Telemetry")

# Initialize Session State Objects
if "stream_context" not in st.session_state:
    st.session_state.stream_context = MultimodalStreamContext(window_size=30)
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
    enable_mic = st.toggle("🎙️ Stream Microphone (Real-Time Audio Prosody)", value=True)

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
        <span style="font-size: 0.76rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8; margin-left: 8px;">{len(st.session_state.session_manager.samples)} samples</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

# Pre-define live text result for fusion
live_text_res = None

# Layout: Left side Video Feed & Actions, Right side Live Telemetry Charts
left_col, right_col = st.columns([5, 6])

# Video & Synthetic Stream Processing
with left_col:
    st.markdown("#### 👁️ Real-Time Visual Viewport")

    if stream_source == "Live Webcam / Camera":
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            
            ctx_container = st.session_state.stream_context

            def video_factory():
                return MultimodalVideoProcessor(context=ctx_container)

            def audio_factory():
                return MultimodalAudioProcessor(context=ctx_container, sample_rate=16000)

            webrtc_ctx = webrtc_streamer(
                key="emotion-sense-live",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=video_factory,
                audio_processor_factory=audio_factory if enable_mic else None,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": bool(enable_mic)},
                sendback_audio=False,
                async_processing=True,
            )
            
            latest_state = ctx_container.get_latest_state()
            if latest_state:
                st.session_state.latest_state = latest_state
                if st.session_state.session_manager.is_recording:
                    st.session_state.session_manager.add_sample(latest_state)
        except Exception as e:
            st.warning(f"WebRTC hardware access unavailable. Running in simulation mode: {e}")
            stream_source = "Synthetic Live Stream Simulator"

    if stream_source == "Synthetic Live Stream Simulator":
        # Interactive Simulator Controls
        st.markdown('<div class="es-panel">', unsafe_allow_html=True)
        st.markdown("<div class='es-section-title'>🎮 Affect Signal Generator</div>", unsafe_allow_html=True)
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            sim_emotion = st.selectbox("Target Simulated Emotion", ["joy", "surprise", "sadness", "anger", "neutral", "fear", "disgust"])
            sim_mic_active = st.checkbox("Simulate Microphone Speech Active", value=True)
        with sim_col2:
            sim_intensity = st.slider("Affect Intensity", 0.1, 1.0, 0.85)
            sim_pitch = st.slider("Simulated Pitch (F0 Hz)", 80, 350, 220 if sim_emotion == "joy" else 150)

        # Generate synthetic vision & audio result
        probs = {emo: 0.05 for emo in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "contempt"]}
        probs[sim_emotion] = float(sim_intensity)
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
            head_pose={"yaw": float(np.random.uniform(-5, 5)), "pitch": float(np.random.uniform(-3, 3)), "roll": 0.0}
        )

        synthetic_acoustics = None
        synthetic_voice = None
        if sim_mic_active:
            synthetic_acoustics = AcousticFeatures(
                pitch_hz=float(sim_pitch),
                pitch_confidence=0.92,
                rms_energy=0.08 * sim_intensity,
                speech_active=True,
                jitter_percent=0.015 if sim_emotion not in ["anger", "fear"] else 0.04,
                shimmer_percent=0.03,
            )
            from src.audio.voice_sentiment import VoiceSentimentClassifier
            classifier = VoiceSentimentClassifier()
            synthetic_voice = classifier.classify_voice_emotion(synthetic_acoustics)
            st.session_state.stream_context.update_voice(synthetic_voice, synthetic_acoustics)

        synthetic_state = st.session_state.stream_context.fusion_engine.fuse(
            vision=synthetic_vision,
            voice=synthetic_voice,
            text=live_text_res
        )
        st.session_state.latest_state = synthetic_state
        st.session_state.stream_context.update_state(synthetic_state, synthetic_vision)
        if st.session_state.session_manager.is_recording:
            st.session_state.session_manager.add_sample(synthetic_state)

        # Render Synthetic Frame Canvas
        canvas = np.zeros((320, 480, 3), dtype=np.uint8)
        canvas[:] = (18, 22, 34)
        # Draw face oval
        cv2.ellipse(canvas, (240, 160), (90, 120), 0, 0, 360, (59, 130, 246), 2)
        # Draw eyes
        cv2.circle(canvas, (205, 130), 8, (14, 165, 233), -1)
        cv2.circle(canvas, (275, 130), 8, (14, 165, 233), -1)
        # Draw mouth based on emotion
        if sim_emotion == "joy":
            cv2.ellipse(canvas, (240, 200), (40, 20), 0, 0, 180, (16, 185, 129), 3)
        elif sim_emotion == "sadness":
            cv2.ellipse(canvas, (240, 220), (35, 15), 0, 180, 360, (59, 130, 246), 3)
        else:
            cv2.line(canvas, (215, 205), (265, 205), (148, 163, 184), 2)

        cv2.putText(canvas, f"GENERATED: {sim_emotion.upper()} ({int(sim_intensity*100)}%)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 245, 249), 1, cv2.LINE_AA)
        
        if sim_mic_active:
            cv2.putText(canvas, f"MIC: {int(sim_pitch)}Hz (ACTIVE)", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1, cv2.LINE_AA)

        st.image(canvas, channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Live Text & Spoken Prompt Input for True Tri-Modal Fusion
    st.markdown("#### 💬 Live Spoken Prompt (Tri-Modal Context)")
    live_prompt = st.text_input(
        "Spoken / Typed Phrase Context",
        placeholder="Type or paste what the speaker is saying...",
        key="live_stream_text_prompt"
    )
    if live_prompt.strip():
        if "text_classifier" not in st.session_state:
            from src.text import TextEmotionClassifier
            st.session_state.text_classifier = TextEmotionClassifier()
        live_text_res = st.session_state.text_classifier.analyze_text(live_prompt)
        st.session_state.stream_context.set_live_text(live_prompt)
        st.markdown(f"""
        <div style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; background: var(--surface-card); border: 1px solid var(--border-color); padding: 4px 10px; border-radius: 4px; margin-bottom: 8px;">
            <b>NLP CONTEXT:</b> <span style="color: #3b82f6;">{live_text_res.dominant_emotion.upper()}</span> ({int(live_text_res.confidence*100)}%)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.session_state.stream_context.set_live_text(None)

    # Action Units Section
    current_state = getattr(st.session_state, "latest_state", None)
    if current_state and current_state.vision:
        st.markdown("<div class='es-section-title'>🧬 Facial Action Units (FACS Spectrum)</div>", unsafe_allow_html=True)
        st.plotly_chart(render_action_units_bar_chart(current_state.vision.action_units), use_container_width=True)

# Telemetry Charts & Affect Dimension
with right_col:
    st.markdown("#### ⚡ Real-Time Telemetry Rack")

    current_state = getattr(st.session_state, "latest_state", None)
    latest_acoustics = st.session_state.stream_context.get_latest_acoustics()
    latest_voice = st.session_state.stream_context.get_latest_voice()

    if current_state:
        render_affect_summary_badge(current_state)
        render_state_indicators(current_state)

        # Acoustic Prosody Mini-Rack
        if latest_acoustics:
            st.markdown("<div class='es-section-title'>🎙️ Acoustic Prosody Telemetry</div>", unsafe_allow_html=True)
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                v_emo = latest_voice.dominant_emotion.upper() if latest_voice else "N/A"
                render_metric_card("Voice Tone", v_emo, delta="Acoustic", color="#8b5cf6")
            with ac2:
                render_metric_card("Pitch F0", f"{latest_acoustics.pitch_hz:.1f} Hz", color="#06b6d4")
            with ac3:
                stress_val = int(latest_voice.vocal_stress_level * 100) if latest_voice else 0
                stress_col = "#ef4444" if stress_val > 60 else ("#f59e0b" if stress_val > 30 else "#10b981")
                render_metric_card("Vocal Stress", f"{stress_val}%", color=stress_col)
            with ac4:
                act_str = "ACTIVE 🎙️" if latest_acoustics.speech_active else "QUIET 🔇"
                render_metric_card("Mic Status", act_str, color="#10b981" if latest_acoustics.speech_active else "#94a3b8")

        tab1, tab2 = st.tabs(["📊 Radar & Circumplex", "📈 Dynamic Timeline"])
        with tab1:
            t1, t2 = st.columns(2)
            with t1:
                st.markdown("<div class='es-section-title'>8-Emotion Polar Radar</div>", unsafe_allow_html=True)
                st.plotly_chart(render_emotion_radar_chart(current_state.probabilities), use_container_width=True)
            with t2:
                st.markdown("<div class='es-section-title'>Russell Circumplex (2D VAD)</div>", unsafe_allow_html=True)
                st.plotly_chart(render_affect_quadrant_chart(current_state.affect), use_container_width=True)

        with tab2:
            st.markdown("<div class='es-section-title'>Affect & Behavioral Telemetry Stream</div>", unsafe_allow_html=True)
            recent_history = st.session_state.stream_context.fusion_engine.get_recent_history()
            st.plotly_chart(render_emotion_timeline_chart(recent_history), use_container_width=True)
    else:
        st.info("Start video stream or simulator above to generate live multimodal telemetry.")

if "pinned_snapshot" in st.session_state:
    st.markdown("---")
    st.markdown("### 📸 Pinned Affect Snapshot")
    snap = st.session_state.pinned_snapshot
    sn1, sn2, sn3, sn4 = st.columns(4)
    with sn1:
        render_metric_card("Snapshot Affect", str(snap.get("dominant_emotion", "neutral")).upper(), delta=f"{int(snap.get('confidence', 0)*100)}% Conf", color="#3b82f6")
    with sn2:
        render_metric_card("Snapshot Valence", f"{snap.get('affect', {}).get('valence', 0):+.2f}", color="#10b981")
    with sn3:
        render_metric_card("Snapshot Arousal", f"{snap.get('affect', {}).get('arousal', 0):+.2f}", color="#f59e0b")
    with sn4:
        import json
        snap_json = json.dumps(snap, indent=2)
        st.download_button("📥 Download Snapshot JSON", data=snap_json, file_name="affect_snapshot.json", mime="application/json", use_container_width=True)
