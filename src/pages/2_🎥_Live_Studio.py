"""Page 2: Real-Time Multimodal Streaming Studio with WebRTC & Telemetry."""

import streamlit as st
import cv2
import numpy as np
import collections
import time

from src.ui.styles import inject_modern_styles
from src.ui.components import (
    render_header,
    render_metric_card,
    render_affect_summary_badge,
    render_state_indicators,
    render_dyadic_summary_card,
    render_participant_badge,
)
from src.ui.charts import (
    render_emotion_radar_chart,
    render_affect_quadrant_chart,
    render_emotion_timeline_chart,
    render_action_units_bar_chart,
    render_dyadic_synchrony_chart,
    render_conversational_dominance_pie,
    render_rapport_gauge,
)
from src.fusion.interaction_dynamics import DyadicInteractionAnalyzer
from src.core.types import (
    VisionEmotionResult,
    FacialActionUnits,
    AcousticFeatures,
    DiarizationResult,
    SpeakerTurn,
)
from src.vision.face_mesh import FaceMeshDetector
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.audio.speech_transcriber import LiveSpeechTranscriber
from src.text.nlp_emotion import TextEmotionClassifier
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
    tracking_mode = st.radio("Tracking Paradigm", ["👤 Single Subject", "👥 Dyadic / Dual Subject"], horizontal=True)
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
is_dyadic_mode = (tracking_mode == "👥 Dyadic / Dual Subject")

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
                return MultimodalVideoProcessor(context=ctx_container, multi_face_mode=is_dyadic_mode)

            def audio_factory():
                return MultimodalAudioProcessor(context=ctx_container, sample_rate=16000)

            webrtc_ctx = webrtc_streamer(
                key=f"emotion-sense-live-{'dyadic' if is_dyadic_mode else 'single'}",
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

            # Process live dyadic metrics when in dyadic mode
            if is_dyadic_mode:
                multi_hist = ctx_container.get_multi_face_history()
                if multi_hist and len(multi_hist) >= 2:
                    samples_a = []
                    samples_b = []
                    for mf in multi_hist:
                        faces_by_id = {f.track_id: f for f in mf.faces}
                        if 0 in faces_by_id and 1 in faces_by_id:
                            f0 = faces_by_id[0]
                            f1 = faces_by_id[1]
                            v0 = f0.vision_result
                            v1 = f1.vision_result
                            samples_a.append({
                                "timestamp": mf.timestamp,
                                "valence": v0.affect.valence if hasattr(v0, "affect") else 0.0,
                                "arousal": v0.affect.arousal if hasattr(v0, "affect") else 0.0,
                                "smile": v0.action_units.lip_corner_puller if v0.action_units else 0.1,
                                "yaw": v0.head_pose.get("yaw", 0.0) if v0.head_pose else 0.0,
                                "dominant_emotion": v0.dominant_emotion,
                                "confidence": v0.confidence,
                                "probabilities": v0.probabilities,
                            })
                            samples_b.append({
                                "timestamp": mf.timestamp,
                                "valence": v1.affect.valence if hasattr(v1, "affect") else 0.0,
                                "arousal": v1.affect.arousal if hasattr(v1, "affect") else 0.0,
                                "smile": v1.action_units.lip_corner_puller if v1.action_units else 0.1,
                                "yaw": v1.head_pose.get("yaw", 0.0) if v1.head_pose else 0.0,
                                "dominant_emotion": v1.dominant_emotion,
                                "confidence": v1.confidence,
                                "probabilities": v1.probabilities,
                            })
                    if len(samples_a) >= 2:
                        analyzer = DyadicInteractionAnalyzer()
                        st.session_state.live_dyadic_metrics = analyzer.analyze(samples_a, samples_b)
                        st.session_state.live_dyadic_samples_a = samples_a
                        st.session_state.live_dyadic_samples_b = samples_b
        except Exception as e:
            st.warning(f"WebRTC hardware access unavailable. Running in simulation mode: {e}")
            stream_source = "Synthetic Live Stream Simulator"

    if stream_source == "Synthetic Live Stream Simulator":
        if not is_dyadic_mode:
            # Single Subject Simulator
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
            cv2.circle(canvas, (205, 130), 8, (14, 165, 233), -1)
            cv2.circle(canvas, (275, 130), 8, (14, 165, 233), -1)
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
        else:
            # Dyadic / Dual-Subject Simulator
            st.markdown('<div class="es-panel">', unsafe_allow_html=True)
            st.markdown("<div class='es-section-title'>👥 Dyadic Interaction Signal Generator</div>", unsafe_allow_html=True)
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown("<b style='color: #3b82f6;'>Participant A (P0)</b>", unsafe_allow_html=True)
                sim_emo_a = st.selectbox("P0 Emotion", ["joy", "surprise", "sadness", "anger", "neutral", "fear", "disgust"], key="sim_emo_a")
                sim_int_a = st.slider("P0 Intensity", 0.1, 1.0, 0.85, key="sim_int_a")
            with p_col2:
                st.markdown("<b style='color: #a855f7;'>Participant B (P1)</b>", unsafe_allow_html=True)
                sim_emo_b = st.selectbox("P1 Emotion", ["joy", "surprise", "neutral", "sadness", "anger", "fear", "disgust"], key="sim_emo_b")
                sim_int_b = st.slider("P1 Intensity", 0.1, 1.0, 0.75, key="sim_int_b")

            sim_dominance_a = st.slider("Floor Share (Participant A %)", 10, 90, 55, format="%d%%") / 100.0

            # Generate synthetic vision for A and B
            def make_synth_vis(emo: str, intensity: float) -> VisionEmotionResult:
                p = {e: 0.05 for e in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "contempt"]}
                p[emo] = float(intensity)
                tot = sum(p.values())
                p = {k: v / tot for k, v in p.items()}
                aus = FacialActionUnits(
                    lip_corner_puller=0.85 if emo == "joy" else 0.1,
                    brow_lowerer=0.75 if emo == "anger" else 0.05,
                    jaw_drop=0.7 if emo in ["surprise", "fear"] else 0.1,
                    lip_corner_depressor=0.65 if emo == "sadness" else 0.05,
                )
                return VisionEmotionResult(
                    face_detected=True,
                    dominant_emotion=emo,
                    confidence=intensity,
                    probabilities=p,
                    action_units=aus,
                    head_pose={"yaw": float(np.random.uniform(-4, 4)), "pitch": 0.0, "roll": 0.0}
                )

            vis_a = make_synth_vis(sim_emo_a, sim_int_a)
            vis_b = make_synth_vis(sim_emo_b, sim_int_b)

            state_a = st.session_state.stream_context.fusion_engine.fuse(vision=vis_a)
            state_b = st.session_state.stream_context.fusion_engine.fuse(vision=vis_b)

            now_t = time.time()
            s_dict_a = {
                "timestamp": now_t,
                "valence": state_a.affect.valence,
                "arousal": state_a.affect.arousal,
                "smile": state_a.vision.action_units.lip_corner_puller if state_a.vision and state_a.vision.action_units else 0.1,
                "yaw": 0.0,
                "dominant_emotion": state_a.dominant_emotion,
                "confidence": state_a.confidence,
                "probabilities": state_a.probabilities,
            }
            s_dict_b = {
                "timestamp": now_t,
                "valence": state_b.affect.valence,
                "arousal": state_b.affect.arousal,
                "smile": state_b.vision.action_units.lip_corner_puller if state_b.vision and state_b.vision.action_units else 0.1,
                "yaw": 0.0,
                "dominant_emotion": state_b.dominant_emotion,
                "confidence": state_b.confidence,
                "probabilities": state_b.probabilities,
            }

            if "sim_dyad_a" not in st.session_state:
                st.session_state.sim_dyad_a = collections.deque(maxlen=40)
            if "sim_dyad_b" not in st.session_state:
                st.session_state.sim_dyad_b = collections.deque(maxlen=40)

            st.session_state.sim_dyad_a.append(s_dict_a)
            st.session_state.sim_dyad_b.append(s_dict_b)
            st.session_state.latest_state = state_a

            # Construct mock DiarizationResult for conversational balance
            mock_diar = DiarizationResult(
                turns=[
                    SpeakerTurn("Speaker_0", 0.0, 10.0 * sim_dominance_a, 10.0 * sim_dominance_a, "Active speech segment"),
                    SpeakerTurn("Speaker_1", 10.0 * sim_dominance_a, 10.0, 10.0 * (1.0 - sim_dominance_a), "Active reply segment"),
                ],
                speakers=["Speaker_0", "Speaker_1"],
                speaker_durations={"Speaker_0": 10.0 * sim_dominance_a, "Speaker_1": 10.0 * (1.0 - sim_dominance_a)},
                dominance_ratios={"Speaker_0": sim_dominance_a, "Speaker_1": 1.0 - sim_dominance_a},
                interruption_count=1,
                total_speech_duration=10.0,
                total_audio_duration=10.0,
            )

            analyzer = DyadicInteractionAnalyzer()
            samples_a_list = list(st.session_state.sim_dyad_a)
            samples_b_list = list(st.session_state.sim_dyad_b)
            dyadic_metrics = analyzer.analyze(samples_a_list, samples_b_list, mock_diar)
            st.session_state.live_dyadic_metrics = dyadic_metrics
            st.session_state.live_dyadic_samples_a = samples_a_list
            st.session_state.live_dyadic_samples_b = samples_b_list

            # Render Dual Face Canvas
            canvas = np.zeros((320, 480, 3), dtype=np.uint8)
            canvas[:] = (18, 22, 34)

            # Draw Face A (P0) on left
            cv2.rectangle(canvas, (30, 40), (220, 290), (59, 130, 246), 2)
            cv2.ellipse(canvas, (125, 150), (60, 85), 0, 0, 360, (59, 130, 246), 2)
            cv2.circle(canvas, (105, 130), 6, (14, 165, 233), -1)
            cv2.circle(canvas, (145, 130), 6, (14, 165, 233), -1)
            if sim_emo_a == "joy":
                cv2.ellipse(canvas, (125, 180), (25, 12), 0, 0, 180, (16, 185, 129), 2)
            else:
                cv2.line(canvas, (110, 180), (140, 180), (148, 163, 184), 2)
            cv2.putText(canvas, f"P0: {sim_emo_a.upper()}", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (59, 130, 246), 2, cv2.LINE_AA)

            # Draw Face B (P1) on right
            cv2.rectangle(canvas, (260, 40), (450, 290), (168, 85, 247), 2)
            cv2.ellipse(canvas, (355, 150), (60, 85), 0, 0, 360, (168, 85, 247), 2)
            cv2.circle(canvas, (335, 130), 6, (192, 132, 252), -1)
            cv2.circle(canvas, (375, 130), 6, (192, 132, 252), -1)
            if sim_emo_b == "joy":
                cv2.ellipse(canvas, (355, 180), (25, 12), 0, 0, 180, (16, 185, 129), 2)
            else:
                cv2.line(canvas, (340, 180), (370, 180), (148, 163, 184), 2)
            cv2.putText(canvas, f"P1: {sim_emo_b.upper()}", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (168, 85, 247), 2, cv2.LINE_AA)

            # Draw Dynamic Synchrony Connection Line
            sync_color = (16, 185, 129) if dyadic_metrics.rapport_score >= 70 else ((245, 158, 11) if dyadic_metrics.rapport_score >= 40 else (239, 68, 68))
            cv2.line(canvas, (185, 150), (295, 150), sync_color, 2)
            cv2.putText(canvas, f"RAPPORT: {dyadic_metrics.rapport_score:.0f}", (190, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.38, sync_color, 1, cv2.LINE_AA)

            st.image(canvas, channels="BGR", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if sim_mic_active:
            cv2.putText(canvas, f"MIC: {int(sim_pitch)}Hz (ACTIVE)", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1, cv2.LINE_AA)

        st.image(canvas, channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Live Speech & Spoken Prompt Ingestion for True Tri-Modal Fusion
    st.markdown("#### 💬 Live Speech & Spoken Text (Tri-Modal Context)")

    # Direct Browser Microphone Ingestion Widget
    with st.expander("🎙️ Direct Microphone Capture & Transcribe", expanded=False):
        st.markdown(
            "<div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 6px;'>"
            "Record a spoken audio snippet to instantly classify vocal prosody and transcribe text into live affect fusion:"
            "</div>",
            unsafe_allow_html=True,
        )
        audio_mic_input = st.audio_input("Record Speech Sample", key="live_studio_audio_input")
        if audio_mic_input is not None:
            raw_audio_bytes = audio_mic_input.getvalue()
            if "live_transcriber" not in st.session_state:
                st.session_state.live_transcriber = LiveSpeechTranscriber()
            if "audio_extractor" not in st.session_state:
                st.session_state.audio_extractor = AcousticProsodyExtractor()
            if "voice_classifier" not in st.session_state:
                st.session_state.voice_classifier = VoiceSentimentClassifier()

            try:
                # Transcribe speech and compute prosody
                trans_res = st.session_state.live_transcriber.transcribe_audio_bytes(raw_audio_bytes)
                if trans_res and trans_res.full_transcript:
                    st.session_state.stream_context.update_transcript(
                        trans_res.full_transcript, trans_res.text_emotion
                    )
                    st.session_state.stream_context.set_live_text(trans_res.full_transcript)
                    st.success(f"Transcribed: \"{trans_res.full_transcript}\"")
                else:
                    st.info("Audio processed. No clear speech detected.")
            except Exception as e:
                st.warning(f"Audio processing notice: {e}")

    live_prompt = st.text_input(
        "Spoken / Typed Phrase Context",
        placeholder="Type or paste what the speaker is saying (or speak via microphone)...",
        key="live_stream_text_prompt",
        value=st.session_state.stream_context.get_latest_transcript() or "",
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

    # Real-Time Spoken Utterance Feed
    transcript_history = st.session_state.stream_context.get_transcript_history()
    if transcript_history:
        st.markdown("<div class='es-section-title'>🗣️ Live Utterances & Speech Affect Feed</div>", unsafe_allow_html=True)
        feed_items = ""
        for item in reversed(transcript_history[-4:]):
            emo = str(item.get("dominant_emotion", "neutral")).upper()
            txt = str(item.get("text", ""))
            conf = int(float(item.get("confidence", 0.0)) * 100)
            feed_items += (
                f"<div style='margin-bottom: 4px; font-size: 0.78rem; font-family: \"JetBrains Mono\", monospace;'>"
                f"<span style='color: #06b6d4;'>●</span> <b>[{emo} {conf}%]</b>: {txt}</div>"
            )
        st.markdown(
            f"<div style='max-height: 120px; overflow-y: auto; background: var(--surface-card); border: 1px solid var(--border-color); border-radius: 6px; padding: 8px; margin-bottom: 12px;'>"
            f"{feed_items}</div>",
            unsafe_allow_html=True,
        )

    # Action Units Section
    current_state = getattr(st.session_state, "latest_state", None)
    if current_state and current_state.vision:
        st.markdown("<div class='es-section-title'>🧬 Facial Action Units (FACS Spectrum)</div>", unsafe_allow_html=True)
        st.plotly_chart(render_action_units_bar_chart(current_state.vision.action_units), use_container_width=True)

# Telemetry Charts & Affect Dimension
with right_col:
    st.markdown("#### ⚡ Real-Time Telemetry Rack")

    if is_dyadic_mode and "live_dyadic_metrics" in st.session_state:
        metrics = st.session_state.live_dyadic_metrics
        samples_a = st.session_state.get("live_dyadic_samples_a", [])
        samples_b = st.session_state.get("live_dyadic_samples_b", [])

        st.markdown("<div class='es-section-title'>👥 Dyadic Interpersonal Interaction Suite</div>", unsafe_allow_html=True)
        render_dyadic_summary_card(
            rapport_score=metrics.rapport_score,
            resonance_category=metrics.resonance_category,
            valence_sync=metrics.valence_synchrony,
            balance=metrics.conversational_balance,
            mimicry=metrics.mimicry_index,
            notes=metrics.summary_notes,
        )

        # Participant status badges
        if samples_a and samples_b:
            last_a = samples_a[-1]
            last_b = samples_b[-1]
            pb_col1, pb_col2 = st.columns(2)
            with pb_col1:
                render_participant_badge(
                    "Participant A (P0)",
                    last_a.get("dominant_emotion", "neutral"),
                    last_a.get("confidence", 0.8),
                    last_a.get("valence", 0.0),
                    last_a.get("arousal", 0.0),
                    color="#3b82f6"
                )
            with pb_col2:
                render_participant_badge(
                    "Participant B (P1)",
                    last_b.get("dominant_emotion", "neutral"),
                    last_b.get("confidence", 0.8),
                    last_b.get("valence", 0.0),
                    last_b.get("arousal", 0.0),
                    color="#a855f7"
                )

        d_tab1, d_tab2, d_tab3 = st.tabs(["⌖ Dynamic Rapport & Balance", "📈 Dynamic Valence Synchrony", "📊 Dual Radar Comparison"])
        with d_tab1:
            g_col1, g_col2 = st.columns([1, 1])
            with g_col1:
                st.plotly_chart(render_rapport_gauge(metrics.rapport_score, metrics.resonance_category), use_container_width=True)
            with g_col2:
                dur_map = {
                    "Participant A": metrics.conversational_balance * 50.0,
                    "Participant B": max(0.0, 100.0 - metrics.conversational_balance * 50.0),
                }
                st.plotly_chart(render_conversational_dominance_pie(dur_map), use_container_width=True)

        with d_tab2:
            times = [s.get("timestamp", idx) for idx, s in enumerate(samples_a)]
            val_a = [s.get("valence", 0.0) for s in samples_a]
            val_b = [s.get("valence", 0.0) for s in samples_b]
            st.plotly_chart(render_dyadic_synchrony_chart(times, val_a, val_b, "Participant A", "Participant B", metrics.valence_synchrony), use_container_width=True)

        with d_tab3:
            if samples_a and samples_b:
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("<div class='es-section-title'>Participant A (P0) Spectrum</div>", unsafe_allow_html=True)
                    st.plotly_chart(render_emotion_radar_chart(samples_a[-1].get("probabilities", {})), use_container_width=True)
                with r2:
                    st.markdown("<div class='es-section-title'>Participant B (P1) Spectrum</div>", unsafe_allow_html=True)
                    st.plotly_chart(render_emotion_radar_chart(samples_b[-1].get("probabilities", {})), use_container_width=True)
    else:
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

            # Spoken / Text Affect Mini-Rack
            if current_state.text:
                st.markdown("<div class='es-section-title'>💬 Spoken & Semantic Affect Telemetry</div>", unsafe_allow_html=True)
                tx1, tx2, tx3, tx4 = st.columns(4)
                with tx1:
                    render_metric_card("Text Emotion", current_state.text.dominant_emotion.upper(), delta="Semantic NLP", color="#3b82f6")
                with tx2:
                    render_metric_card("Valence", f"{current_state.text.affect.valence:+.2f}", color="#10b981")
                with tx3:
                    render_metric_card("Arousal", f"{current_state.text.affect.arousal:+.2f}", color="#f59e0b")
                with tx4:
                    render_metric_card("Confidence", f"{int(current_state.text.confidence * 100)}%", color="#06b6d4")

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
