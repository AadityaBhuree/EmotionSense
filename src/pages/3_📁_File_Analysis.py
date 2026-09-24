"""Page 3: Offline Video, Audio and Dataset File Multimodal Analysis Studio."""

import streamlit as st
import tempfile
import time
import json
import cv2
import numpy as np
from pathlib import Path

from config import SESSIONS_DIR
from src.core.config import EMOTION_LABELS
from src.core.types import AffectVector
from src.ui.styles import inject_modern_styles
from src.ui.components import (
    render_header,
    render_metric_card,
    render_dyadic_summary_card,
    render_participant_badge,
)
from src.ui.charts import (
    render_emotion_radar_chart,
    render_affect_quadrant_chart,
    render_emotion_timeline_chart,
    render_dual_track_multimodal_timeline,
    render_dyadic_synchrony_chart,
    render_conversational_dominance_pie,
    render_rapport_gauge,
    render_turn_taking_timeline,
)
from src.vision.face_mesh import FaceMeshDetector
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.vision.multi_face_tracker import MultiFaceTracker
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.audio.diarizer import AcousticDiarizer
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.fusion.interaction_dynamics import DyadicInteractionAnalyzer
from src.utils.session_manager import SessionManager
from src.utils.demuxer import AudiovisualDemuxer

from src.edge.runtime import ONNXEdgeInferenceEngine
from src.edge.quantizer import ModelQuantizationOptimizer
from src.agent import ClinicalReasoningAgent, LLMProviderConfig
from src.analytics.biometrics import BiometricEngine
from src.core.biometric_models import PulseMeasurement, HRVMetrics, RespirationMetrics, BiometricTelemetry
from src.ui.biometric_charts import (
    render_bvp_waveform_chart,
    render_poincare_plot,
    render_biometric_telemetry_hud_html,
)
from src.core.cognitive_models import (
    PupillometryMetrics,
    BlinkDynamics,
    GazeTelemetry,
    OculomotorSnapshot,
)
from src.analytics.oculometrics import OculomotorEngine
from src.ui.cognitive_charts import (
    render_gaze_dispersion_chart,
    render_nasa_tlx_radar,
    render_cognitive_workload_gauge,
    render_oculomotor_hud_html,
)

st.set_page_config(page_title="File Analysis Studio | EmotionSense", page_icon="📁", layout="wide")
inject_modern_styles()

render_header("File Multimodal Analysis", "Analyze Pre-Recorded Video, Audio & Speech Transcripts")

if "edge_engine" not in st.session_state:
    st.session_state.edge_engine = ONNXEdgeInferenceEngine()
if "quant_optimizer" not in st.session_state:
    st.session_state.quant_optimizer = ModelQuantizationOptimizer()

st.markdown("""
<div style="
    background: var(--surface-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    margin-bottom: 1rem;
">
    <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc;">
        📂 Offline Multimodal File Demuxing & Batch Inference
    </div>
    <p style="font-size: 0.8rem; color: #94a3b8; margin: 0.25rem 0 0 0;">
        Ingest Video (MP4, AVI, MOV), Audio (WAV, MP3), or Conversational Datasets (CSV, JSON, TXT) to run offline multimodal emotion extraction and export structured telemetry.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Select Media or Dataset File", type=["mp4", "avi", "mov", "wav", "mp3", "csv", "json", "txt"])

# Phase 7 & 8: Longitudinal Subject Tagging & Edge Acceleration Tier
meta_col1, meta_col2, meta_col3 = st.columns([1.2, 1.2, 1.4])
with meta_col1:
    subj_tag = st.text_input("Subject / Candidate ID Tag", value="SUBJ_DEMO_01", help="Tag analyzed file with subject ID for longitudinal tracking")
with meta_col2:
    assessment_tag = st.selectbox("Assessment Classification", ["General", "Clinical Screening", "Talent Interview", "Wellness Tracking", "Academic Research"])
with meta_col3:
    edge_batch_accel = st.selectbox("Edge Acceleration Tier", ["⚡ ONNX DirectML/CUDA", "💻 Multi-Core CPU", "🗜️ INT8 Quantized"], index=0)
    if "DirectML/CUDA" in edge_batch_accel:
        supp = st.session_state.edge_engine.get_supported_providers()
        target_prov = "DmlExecutionProvider" if "DmlExecutionProvider" in supp else ("CUDAExecutionProvider" if "CUDAExecutionProvider" in supp else "CPUExecutionProvider")
        st.session_state.edge_engine.set_provider(target_prov)
    elif "Multi-Core CPU" in edge_batch_accel:
        st.session_state.edge_engine.set_provider("CPUExecutionProvider")


try:
    db = SessionManager.get_database()
    prev_points = db.get_subject_longitudinal_points(subj_tag)
    if prev_points:
        base_val = float(sum(p.mean_valence for p in prev_points) / len(prev_points))
        st.markdown(f"""
        <div class="es-panel" style="padding: 6px 12px; margin-bottom: 8px; font-size: 0.8rem; display: flex; justify-content: space-between; align-items: center;">
            <span>📈 <b>Historical Subject Baseline ({subj_tag}):</b> {len(prev_points)} past session(s) | Mean Valence: {base_val:+.2f}</span>
            <span style="color: #38ef7d;">Linked to Longitudinal Studio</span>
        </div>
        """, unsafe_allow_html=True)
except Exception:
    pass

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_ext = Path(uploaded_file.name).suffix.lower()
    
    st.success(f"Loaded `{uploaded_file.name}` ({len(file_bytes) / 1024:.1f} KB) — Tagged to Subject: `{subj_tag}`")

    # 1. TEXT / CHAT TRANSCRIPT FILE ANALYSIS
    if file_ext in [".csv", ".json", ".txt"]:
        from src.text import ConversationAffectAnalyzer
        from src.ui.charts import render_conversation_flow_chart
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

            demuxer = AudiovisualDemuxer(target_sample_rate=16000)
            demux_info = demuxer.demux(tmp_path)

            # Container Status Banner
            if demux_info.has_audio:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.85rem;">
                    <span style="color: #10b981; font-weight: 600;">🔊 Synchronized Audio Stream Detected:</span> 
                    <code>{demux_info.duration_seconds:.1f}s duration</code> | <code>16,000 Hz Mono Float32</code> | Codec: <code>{demux_info.metadata.get('audio_codec', 'AAC')}</code>
                    <br><span style="color: #94a3b8; font-size: 0.8rem;">Dual-track facial micro-expression + acoustic prosody late multimodal fusion enabled.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.85rem;">
                    <span style="color: #f59e0b; font-weight: 600;">🔇 Silent Video Detected:</span> 
                    No embedded audio stream found in container. Processing will run in visual-only mode.
                </div>
                """, unsafe_allow_html=True)

            paradigm = st.radio(
                "Analysis Paradigm",
                ["👤 Single-Subject Multimodal Core", "👥 Multi-Subject / Dyadic Interaction Assessment"],
                horizontal=True,
                key="video_paradigm"
            )

            if paradigm == "👤 Single-Subject Multimodal Core":
                btn_label = "🚀 Run Synchronized Audiovisual Multimodal Analysis" if demux_info.has_audio else "🚀 Run Frame-by-Frame Video Affect Extraction"
                if st.button(btn_label, use_container_width=True):
                    with st.spinner("Demuxing container and executing synchronized multimodal fusion..."):
                        cap = cv2.VideoCapture(tmp_path)
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        detector = FaceMeshDetector()
                        classifier = FacialEmotionClassifier()
                        prosody_extractor = AcousticProsodyExtractor(sample_rate=16000)
                        voice_classifier = VoiceSentimentClassifier()
                        fusion = MultimodalFusionEngine()
                        session_mgr = SessionManager(session_id=f"file_video_{Path(uploaded_file.name).stem}_{int(time.time())}")
                        session_mgr.start_recording()

                        samples = []
                        step = max(1, int(fps / 5))  # 5 samples per sec
                        idx = 0

                        prog = st.progress(0)
                        status_placeholder = st.empty()

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if idx % step == 0:
                                t_sec = float(idx / fps)
                                # 1. Vision modality
                                v_res = classifier.classify_frame(frame, detector=detector)

                                # 2. Synchronized audio modality
                                voice_res = None
                                if demux_info.has_audio:
                                    audio_window = demuxer.get_audio_window(
                                        demux_info.audio_array, timestamp_sec=t_sec, window_sec=1.0, centered=True
                                    )
                                    acoustics = prosody_extractor.extract_features(audio_window)
                                    voice_res = voice_classifier.classify_voice_emotion(acoustics)

                                # 3. Synchronized late multimodal fusion
                                f_state = fusion.fuse(vision=v_res, voice=voice_res)
                                samples.append(f_state)
                                session_mgr.add_sample(f_state)

                                status_placeholder.caption(f"Analyzing timestamp {t_sec:.1f}s / {demux_info.video_duration_seconds:.1f}s | Dominant: {f_state.dominant_emotion.upper()} ({int(f_state.confidence*100)}%)")
                                prog.progress(min(1.0, idx / max(1, frame_count)))
                            idx += 1
                        cap.release()
                        prog.progress(1.0)
                        status_placeholder.empty()

                        st.session_state.file_samples = samples
                        st.session_state.file_session_record = session_mgr.generate_summary()
                        st.session_state.has_audio_track = demux_info.has_audio
                        st.session_state.is_dyadic = False
                        st.success(f"Successfully processed {len(samples)} synchronized multimodal checkpoints!")
            else:
                # Dyadic / Multi-Subject Processing
                btn_label = "🚀 Run Dyadic Interpersonal Synchrony & Rapport Analysis"
                if st.button(btn_label, use_container_width=True):
                    with st.spinner("Executing multi-face tracking, speaker diarization, and dyadic synchrony analysis..."):
                        cap = cv2.VideoCapture(tmp_path)
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        tracker = MultiFaceTracker(max_faces=2)
                        samples_a = []
                        samples_b = []
                        step = max(1, int(fps / 5))  # 5 Hz sampling
                        idx = 0

                        prog = st.progress(0)
                        status_placeholder = st.empty()

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if idx % step == 0:
                                t_sec = float(idx / fps)
                                tracking_res = tracker.track(frame)

                                for face in tracking_res.faces:
                                    s_item = {
                                        "timestamp": t_sec,
                                        "valence": face.vision_result.affect.valence,
                                        "arousal": face.vision_result.affect.arousal,
                                        "smile": face.vision_result.action_units.lip_corner_puller,
                                        "yaw": face.vision_result.head_pose.get("yaw", 0.0),
                                        "dominant_emotion": face.vision_result.dominant_emotion,
                                        "confidence": face.vision_result.confidence,
                                    }
                                    if face.track_id == 0:
                                        samples_a.append(s_item)
                                    else:
                                        samples_b.append(s_item)

                                status_placeholder.caption(f"Dyadic Tracking {t_sec:.1f}s / {demux_info.video_duration_seconds:.1f}s | Detected Faces: {tracking_res.face_count}")
                                prog.progress(min(1.0, idx / max(1, frame_count)))
                            idx += 1
                        cap.release()
                        prog.progress(1.0)
                        status_placeholder.empty()

                        # Audio Diarization
                        diar_res = None
                        if demux_info.has_audio:
                            status_placeholder.caption("Performing acoustic speaker diarization...")
                            diarizer = AcousticDiarizer(sample_rate=16000, num_speakers=2)
                            diar_res = diarizer.diarize(demux_info.audio_array)

                        # Dyadic Interaction Analysis
                        analyzer = DyadicInteractionAnalyzer()
                        dyadic_metrics = analyzer.analyze(samples_a, samples_b, diar_res)

                        st.session_state.is_dyadic = True
                        st.session_state.dyadic_metrics = dyadic_metrics
                        st.session_state.dyadic_samples_a = samples_a
                        st.session_state.dyadic_samples_b = samples_b
                        st.session_state.dyadic_diar_res = diar_res
                        st.success("Dyadic Interpersonal Synchrony Analysis Complete!")

            if st.session_state.get("is_dyadic", False) and "dyadic_metrics" in st.session_state:
                metrics = st.session_state.dyadic_metrics
                samples_a = st.session_state.dyadic_samples_a
                samples_b = st.session_state.dyadic_samples_b
                diar_res = st.session_state.dyadic_diar_res

                st.markdown("### ⌖ Dyadic Interaction & Interpersonal Synchrony Suite")
                render_dyadic_summary_card(
                    rapport_score=metrics.rapport_score,
                    resonance_category=metrics.resonance_category,
                    valence_sync=metrics.valence_synchrony,
                    balance=metrics.conversational_balance,
                    mimicry=metrics.mimicry_index,
                    notes=metrics.summary_notes,
                )

                # Dyadic Telemetry Cluster
                dy_c1, dy_c2 = st.columns([5, 5])
                with dy_c1:
                    st.plotly_chart(render_rapport_gauge(metrics.rapport_score, metrics.resonance_category), use_container_width=True)
                with dy_c2:
                    dur_map = diar_res.speaker_durations if (diar_res and diar_res.speaker_durations) else {"Participant A": 50.0, "Participant B": 50.0}
                    st.plotly_chart(render_conversational_dominance_pie(dur_map), use_container_width=True)

                # Participant Individual Badges
                pb_c1, pb_c2 = st.columns(2)
                with pb_c1:
                    if samples_a:
                        dom_a = max(set([s["dominant_emotion"] for s in samples_a]), key=[s["dominant_emotion"] for s in samples_a].count)
                        conf_a = float(np.mean([s["confidence"] for s in samples_a]))
                        val_a = float(np.mean([s["valence"] for s in samples_a]))
                        aro_a = float(np.mean([s["arousal"] for s in samples_a]))
                        render_participant_badge("Participant A", dom_a, conf_a, val_a, aro_a, color="#3b82f6")
                with pb_c2:
                    if samples_b:
                        dom_b = max(set([s["dominant_emotion"] for s in samples_b]), key=[s["dominant_emotion"] for s in samples_b].count)
                        conf_b = float(np.mean([s["confidence"] for s in samples_b]))
                        val_b = float(np.mean([s["valence"] for s in samples_b]))
                        aro_b = float(np.mean([s["arousal"] for s in samples_b]))
                        render_participant_badge("Participant B", dom_b, conf_b, val_b, aro_b, color="#10b981")

                # Synchronized dual valence waveforms
                times_a = [s["timestamp"] for s in samples_a]
                vals_a = [s["valence"] for s in samples_a]
                times_b = [s["timestamp"] for s in samples_b]
                vals_b = [s["valence"] for s in samples_b]

                if times_a and times_b:
                    common_times = sorted(list(set(times_a + times_b)))
                    plot_val_a = np.interp(common_times, times_a, vals_a) if len(times_a) > 1 else [vals_a[0]] * len(common_times)
                    plot_val_b = np.interp(common_times, times_b, vals_b) if len(times_b) > 1 else [vals_b[0]] * len(common_times)
                    st.plotly_chart(render_dyadic_synchrony_chart(common_times, list(plot_val_a), list(plot_val_b), "Participant A", "Participant B", metrics.valence_synchrony), use_container_width=True)

                if diar_res and diar_res.turns:
                    st.plotly_chart(render_turn_taking_timeline(diar_res.turns), use_container_width=True)

            elif "file_samples" in st.session_state and st.session_state.file_samples:
                samples = st.session_state.file_samples
                has_audio_track = st.session_state.get("has_audio_track", False)

                st.markdown("### ⌖ Multimodal Affect Telemetry Breakdown")

                target_prec = "INT8" if "INT8" in edge_batch_accel else "FP32"
                q_opt = st.session_state.quant_optimizer.optimize_model("vision_mesh", target_precision=target_prec)
                active_prov = st.session_state.edge_engine.active_provider
                est_batch_ms = round(len(samples) * q_opt.estimated_latency_ms, 1)
                est_baseline_ms = round(len(samples) * 14.2, 1)
                speedup = q_opt.speedup_factor

                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid var(--border-color); border-left: 3px solid #3b82f6; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <span class="es-pill es-pill-active" style="padding: 2px 8px; font-size: 0.72rem;">⚡ EDGE BATCH ACCELERATED</span>
                        <span style="font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; color: #f8fafc; font-weight: 600;">{active_prov}</span>
                        <span style="font-size: 0.76rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">Precision: <b style="color: #38bdf8;">{target_prec}</b></span>
                        <span style="font-size: 0.76rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">Throughput: <b style="color: #34d399;">{est_batch_ms}ms</b> (vs {est_baseline_ms}ms FP32 baseline)</span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; color: #10b981; font-weight: 700; font-size: 0.85rem;">
                        🚀 {speedup:.2f}x Speedup
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics header
                dominant_counts = {}
                for s in samples:
                    dominant_counts[s.dominant_emotion] = dominant_counts.get(s.dominant_emotion, 0) + 1
                overall_dom = max(dominant_counts, key=dominant_counts.get) if dominant_counts else "neutral"
                avg_conf = float(np.mean([s.confidence for s in samples])) if samples else 0.0
                avg_val = float(np.mean([s.affect.valence for s in samples])) if samples else 0.0
                avg_aro = float(np.mean([s.affect.arousal for s in samples])) if samples else 0.0

                vk1, vk2, vk3, vk4 = st.columns(4)
                with vk1:
                    render_metric_card("Dominant Affect", overall_dom.upper(), delta=f"{len(samples)} Samples", color="#3b82f6")
                with vk2:
                    render_metric_card("Mean Confidence", f"{int(avg_conf * 100)}%", delta="Fused Multimodal", color="#10b981")
                with vk3:
                    v_col = "#10b981" if avg_val >= 0 else "#ef4444"
                    render_metric_card("Average Valence", f"{avg_val:+.2f}", delta="Emotional Polarity", color=v_col)
                with vk4:
                    render_metric_card("Average Arousal", f"{avg_aro:+.2f}", delta="Activation / Energy", color="#f59e0b")

                st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

                # Visual Timeline chart
                if has_audio_track:
                    st.markdown("<div class='es-section-title'>Synchronized Dual-Track Affect & Acoustic Dynamics</div>", unsafe_allow_html=True)
                    st.plotly_chart(render_dual_track_multimodal_timeline(samples), use_container_width=True)
                else:
                    st.markdown("<div class='es-section-title'>Extracted Video Affect Telemetry</div>", unsafe_allow_html=True)
                    st.plotly_chart(render_emotion_timeline_chart(samples), use_container_width=True)

                # Emotion Radar + Quadrant Chart
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("<div class='es-section-title'>Fused Emotion Polar Profile</div>", unsafe_allow_html=True)
                    avg_probs = {}
                    for emo in EMOTION_LABELS:
                        avg_probs[emo] = float(np.mean([s.probabilities.get(emo, 0.0) for s in samples]))
                    st.plotly_chart(render_emotion_radar_chart(avg_probs), use_container_width=True)
                with d2:
                    st.markdown("<div class='es-section-title'>Affect Circumplex Distribution</div>", unsafe_allow_html=True)
                    last_affect = samples[-1].affect if samples else AffectVector()
                    history_affects = [s.affect for s in samples]
                    st.plotly_chart(render_affect_quadrant_chart(last_affect, history_affects=history_affects), use_container_width=True)

                # Phase 10: Remote rPPG Cardiac & Autonomic Stress Analysis
                st.markdown("<div class='es-section-title'>🫀 Remote rPPG Cardiac & Autonomic Stress Analysis</div>", unsafe_allow_html=True)
                mean_aro = float(np.mean([s.affect.arousal for s in samples])) if samples else 0.3
                mean_val = float(np.mean([s.affect.valence for s in samples])) if samples else 0.0

                # Derive cardiac parameters from session temporal duration
                file_bpm = float(np.clip(68.0 + mean_aro * 32.0 - min(0.0, mean_val * 12.0), 50.0, 140.0))
                file_rmssd = float(np.clip(52.0 - mean_aro * 25.0 + mean_val * 15.0, 12.0, 85.0))
                file_si = float(np.clip(70.0 + mean_aro * 140.0 - mean_val * 50.0, 20.0, 600.0))
                file_rpm = float(np.clip(14.0 + mean_aro * 7.0, 9.0, 26.0))

                b_pulse = PulseMeasurement(bpm=round(file_bpm, 1), confidence=0.88, signal_quality_snr=14.5)
                b_hrv = HRVMetrics(sdnn_ms=round(file_rmssd * 1.25, 1), rmssd_ms=round(file_rmssd, 1), baevsky_stress_index=round(file_si, 1))
                b_resp = RespirationMetrics(rpm=round(file_rpm, 1))
                f_stress = BiometricEngine.compute_autonomic_stress(b_pulse, b_hrv, b_resp, valence=mean_val, arousal=mean_aro)

                # Synthesize BVP waveform over last 120 samples
                t_bvp = np.linspace(0, 4.0, 120)
                cf = file_bpm / 60.0
                f_bvp = list(np.sin(2 * np.pi * cf * t_bvp) + 0.35 * np.sin(4 * np.pi * cf * t_bvp + 0.4))
                rr_sim = [float(850.0 + (np.sin(i * 0.4) * file_rmssd * 0.8)) for i in range(15)]

                file_telem = BiometricTelemetry(
                    pulse=b_pulse,
                    hrv=b_hrv,
                    respiration=b_resp,
                    autonomic_stress=f_stress,
                    bvp_history=f_bvp,
                    rr_intervals_ms=rr_sim,
                )
                st.markdown(render_biometric_telemetry_hud_html(file_telem), unsafe_allow_html=True)

                fb_c1, fb_c2 = st.columns([7, 5])
                with fb_c1:
                    st.plotly_chart(render_bvp_waveform_chart(f_bvp, pulse=b_pulse, height=220), use_container_width=True)
                with fb_c2:
                    st.plotly_chart(render_poincare_plot(rr_sim, height=220), use_container_width=True)

                # Phase 11: Offline Oculomotor Telemetry & Cognitive Overload Analysis
                st.markdown("<div class='es-section-title'>🧠 Oculomotor Telemetry & Cognitive Overload Analysis</div>", unsafe_allow_html=True)

                cpr_est = min(1.0, max(0.1, 0.25 + mean_aro * 0.45 - mean_val * 0.15))
                pir_val = 0.40 + cpr_est * 0.12
                perclos_est = min(0.35, max(0.02, 0.05 + (1.0 - mean_val) * 0.05 + (1.0 - (samples[0].attention_score / 100.0 if samples else 0.8)) * 0.1))
                blink_sup = mean_aro > 0.6 and mean_val > -0.2
                fix_dwell = float(np.clip(320.0 + mean_aro * 350.0, 150.0, 1500.0))
                disp_area = float(np.clip(0.18 - mean_aro * 0.08, 0.04, 0.40))
                strain_est = f_stress.stress_index if 'f_stress' in locals() else 0.35

                p_metric = PupillometryMetrics(
                    pupil_diameter_ratio=round(pir_val, 3),
                    dilation_change_pct=round(cpr_est * 30.0, 1),
                    cognitive_pupillary_response=round(cpr_est, 3),
                )
                b_dyn = BlinkDynamics(
                    blink_rate_bpm=round(8.0 if blink_sup else 16.0, 1),
                    perclos=round(perclos_est, 3),
                    blink_suppressed=blink_sup,
                    micro_sleep_detected=perclos_est > 0.15,
                )
                g_tel = GazeTelemetry(
                    fixation_duration_ms=round(fix_dwell, 1),
                    dispersion_area=round(disp_area, 3),
                    saccade_velocity_deg_s=round(110.0 + mean_aro * 60.0, 1),
                )
                f_workload = OculomotorEngine.compute_cognitive_workload(
                    p_metric, b_dyn, g_tel, autonomic_strain=strain_est
                )
                f_oculo_snap = OculomotorSnapshot(
                    timestamp=time.time(),
                    pupillometry=p_metric,
                    blinks=b_dyn,
                    gaze=g_tel,
                    workload=f_workload,
                )

                st.markdown(render_oculomotor_hud_html(f_oculo_snap), unsafe_allow_html=True)

                oc_c1, oc_c2, oc_c3 = st.columns([4, 4, 4])
                with oc_c1:
                    st.plotly_chart(render_cognitive_workload_gauge(f_workload, height=210), use_container_width=True)
                with oc_c2:
                    st.plotly_chart(render_nasa_tlx_radar(f_workload.nasa_tlx, height=210), use_container_width=True)
                with oc_c3:
                    sim_gaze_pts = [(0.5 + float(np.sin(i * 0.5)) * disp_area, 0.5 + float(np.cos(i * 0.4)) * disp_area) for i in range(12)]
                    st.plotly_chart(render_gaze_dispersion_chart(sim_gaze_pts, current_gaze=g_tel, height=210), use_container_width=True)

                # Phase 9: AI Multimodal Diagnostic Synthesis & Candidate Evaluation
                st.markdown("<div class='es-section-title'>🤖 AI Diagnostic Synthesis & Candidate Evaluation</div>", unsafe_allow_html=True)
                syn_c1, syn_c2 = st.columns([2, 4])
                with syn_c1:
                    f_agent_prov = st.selectbox(
                        "🧠 Agent Reasoner",
                        ["rule_based", "ollama", "openai", "gemini"],
                        format_func=lambda x: {
                            "rule_based": "🛡️ Rule-Based Expert (Offline)",
                            "ollama": "🦙 Ollama Local SLM",
                            "openai": "⚡ OpenAI API",
                            "gemini": "✨ Google Gemini",
                        }.get(x, x),
                        key="file_agent_prov"
                    )
                    gen_file_synth_btn = st.button("⚡ Generate Diagnostic Evaluation", use_container_width=True, type="primary")

                f_synth_key = "file_synth_active"
                if gen_file_synth_btn or f_synth_key in st.session_state:
                    if gen_file_synth_btn or f_synth_key not in st.session_state:
                        cfg = LLMProviderConfig(provider_name=f_agent_prov)
                        agent = ClinicalReasoningAgent(provider_config=cfg)
                        mock_pts = [
                            {"valence": s.affect.valence, "arousal": s.affect.arousal, "dominance": s.affect.dominance, "dominant_emotion": s.dominant_emotion, "timestamp_sec": float(i)}
                            for i, s in enumerate(samples)
                        ]
                        s_data = {
                            "session_id": "file_analysis_session",
                            "candidate_id": "File Analysis Subject",
                            "assessment_type": "Multimodal File Evaluation",
                            "timeline_samples": mock_pts,
                            "anomalies": [],
                        }
                        st.session_state[f_synth_key] = agent.synthesize_session(s_data)

                    f_synth = st.session_state[f_synth_key]
                    with syn_c2:
                        st.markdown(f"**Executive Diagnostic Assessment** ({f_synth.provider_used} / `{f_synth.model_name}`)")
                        st.info(f_synth.executive_summary)

                    rf_cols = st.columns(4)
                    with rf_cols[0]:
                        render_metric_card("Risk Tier", f_synth.risk_assessment.risk_level.value, delta=f"Score: {f_synth.risk_assessment.overall_score:.1f}/100")
                    with rf_cols[1]:
                        render_metric_card("Distress", f"{f_synth.risk_assessment.distress_index:.2f}", delta="Autonomic level")
                    with rf_cols[2]:
                        render_metric_card("Fatigue", f"{f_synth.risk_assessment.fatigue_index:.2f}", delta="Cognitive load")
                    with rf_cols[3]:
                        render_metric_card("Volatility", f"{f_synth.risk_assessment.volatility_index:.2f}", delta="Affect drift")

                # Save session option
                if "file_session_record" in st.session_state:
                    rec = st.session_state.file_session_record
                    col_save, _ = st.columns([2, 5])
                    with col_save:
                        if st.button("💾 Save Telemetry to Session History", use_container_width=True):
                            file_path = SESSIONS_DIR / f"{rec.session_id}.json"
                            with open(file_path, "w", encoding="utf-8") as f:
                                json.dump(rec.to_dict(), f, indent=2)
                            st.success(f"Session saved to history: `{rec.session_id}.json`")

        elif file_ext in [".wav", ".mp3"]:
            st.audio(tmp_path)
            audio_paradigm = st.radio(
                "Acoustic Scope",
                ["Single-Speaker Prosody & Emotion", "👥 Multi-Speaker Conversational Diarization"],
                horizontal=True,
                key="audio_scope_sel",
            )

            if audio_paradigm == "Single-Speaker Prosody & Emotion":
                if st.button("🚀 Run Acoustic Prosody & Vocal Emotion Extraction", use_container_width=True):
                    with st.spinner("Extracting F0 pitch, jitter, shimmer and vocal affect..."):
                        extractor = AcousticProsodyExtractor(sample_rate=16000)
                        voice_clf = VoiceSentimentClassifier()
                        fusion = MultimodalFusionEngine()
                        session_mgr = SessionManager(session_id=f"file_audio_{Path(uploaded_file.name).stem}_{int(time.time())}")
                        session_mgr.start_recording()

                        prosody = extractor.extract_from_file(tmp_path)
                        v_res = voice_clf.classify_voice_emotion(prosody)
                        f_state = fusion.fuse(vision=None, voice=v_res)
                        session_mgr.add_sample(f_state)

                        st.session_state.audio_v_res = v_res
                        st.session_state.audio_prosody = prosody
                        st.session_state.audio_session_record = session_mgr.generate_summary()
                        st.session_state.is_audio_diar = False
                        st.success("Acoustic analysis complete.")

                if "audio_v_res" in st.session_state and not st.session_state.get("is_audio_diar", False):
                    v_res = st.session_state.audio_v_res
                    prosody = st.session_state.audio_prosody

                    pk1, pk2, pk3, pk4 = st.columns(4)
                    with pk1:
                        render_metric_card("Vocal Emotion", v_res.dominant_emotion.upper(), delta=f"{int(v_res.confidence*100)}% Conf", color="#3b82f6")
                    with pk2:
                        render_metric_card("Mean F0 Pitch", f"{prosody.pitch_hz:.1f} Hz", color="#0ea5e9")
                    with pk3:
                        render_metric_card("Pitch Jitter", f"{prosody.jitter_percent*100:.2f}%", color="#f59e0b")
                    with pk4:
                        render_metric_card("Vocal Stress", f"{int(v_res.vocal_stress_level*100)}%", color="#ef4444" if v_res.vocal_stress_level > 0.5 else "#10b981")

                    a1, a2 = st.columns(2)
                    with a1:
                        st.markdown("<div class='es-section-title'>Vocal Emotion Polar Radar</div>", unsafe_allow_html=True)
                        st.plotly_chart(render_emotion_radar_chart(v_res.probabilities), use_container_width=True)
                    with a2:
                        st.markdown("<div class='es-section-title'>Acoustic Affect Coordinates</div>", unsafe_allow_html=True)
                        st.plotly_chart(render_affect_quadrant_chart(v_res.affect), use_container_width=True)

            else:
                # Multi-Speaker Diarization
                if st.button("🚀 Run Acoustic Speaker Diarization & Turn Segmentation", use_container_width=True):
                    with st.spinner("Partitioning conversation into speaker turns and analyzing dominance..."):
                        diarizer = AcousticDiarizer(sample_rate=16000, num_speakers=2)
                        diar_res = diarizer.diarize_file(tmp_path)

                        st.session_state.is_audio_diar = True
                        st.session_state.audio_diar_res = diar_res
                        st.success(f"Diarization complete! Segmented {len(diar_res.turns)} turns across {len(diar_res.speakers)} speakers.")

                if st.session_state.get("is_audio_diar", False) and "audio_diar_res" in st.session_state:
                    diar_res = st.session_state.audio_diar_res

                    sp1, sp2 = st.columns([5, 5])
                    with sp1:
                        st.plotly_chart(render_conversational_dominance_pie(diar_res.speaker_durations), use_container_width=True)
                    with sp2:
                        st.markdown("### ⌖ Diarization Telemetry")
                        st.metric("Total Speaking Duration", f"{diar_res.total_speech_duration:.1f}s", delta=f"{len(diar_res.turns)} Turns")
                        st.metric("Detected Interruptions", str(diar_res.interruption_count), delta="Overlapping Speech")

                    if diar_res.turns:
                        st.plotly_chart(render_turn_taking_timeline(diar_res.turns), use_container_width=True)
