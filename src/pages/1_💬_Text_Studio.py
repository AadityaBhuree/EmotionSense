"""Page 1: Dedicated Text Message & Conversational Affective Intelligence Studio."""

import streamlit as st
import time
import pandas as pd
import json

from config import THEME_COLORS, EMOTION_COLORS
from src.ui.styles import inject_modern_styles
from src.ui.components import (
    render_header,
    render_metric_card,
    render_chat_bubble,
    render_token_explainability_box,
    render_empathy_advice_card,
)
from src.ui.charts import (
    render_emotion_radar_chart,
    render_affect_quadrant_chart,
    render_conversation_flow_chart,
    render_emotion_distribution_pie,
    render_emotion_horizontal_bars,
)
from src.text import TextEmotionClassifier, ConversationAffectAnalyzer, HybridEmotionClassifier
from src.audio.speech_transcriber import LiveSpeechTranscriber
from src.audio.prosody import AcousticProsodyExtractor
from src.audio.voice_sentiment import VoiceSentimentClassifier
from src.utils.session_manager import SessionManager

# Page Configuration
st.set_page_config(page_title="Text Emotion Studio | EmotionSense", page_icon="💬", layout="wide")
inject_modern_styles()

render_header("Text Emotion & Dialogue Studio", "Deep Affective NLP, Conversational Trajectory & Spoken Speech Ingestion")

# Initialize Session State
if "hybrid_classifier" not in st.session_state:
    st.session_state.hybrid_classifier = HybridEmotionClassifier(mode="hybrid")
if "conversation_analyzer" not in st.session_state:
    st.session_state.conversation_analyzer = ConversationAffectAnalyzer(st.session_state.hybrid_classifier)
if "speech_transcriber" not in st.session_state:
    st.session_state.speech_transcriber = LiveSpeechTranscriber(st.session_state.hybrid_classifier.lexical_clf)
if "prosody_extractor" not in st.session_state:
    st.session_state.prosody_extractor = AcousticProsodyExtractor()
if "voice_classifier" not in st.session_state:
    st.session_state.voice_classifier = VoiceSentimentClassifier()
if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()
if "single_text_input" not in st.session_state:
    st.session_state.single_text_input = "I am absolutely thrilled and excited about our new project launch!! The results are fantastic! 🚀🎉"

clf = st.session_state.hybrid_classifier
conv = st.session_state.conversation_analyzer
transcriber = st.session_state.speech_transcriber
prosody_ext = st.session_state.prosody_extractor
voice_clf = st.session_state.voice_classifier


# Controls Bar: Studio Mode & NLP Engine Selector
ctl1, ctl2 = st.columns([6, 4])
with ctl1:
    mode = st.radio(
        "Studio Analysis Mode",
        ["📝 Single Message & Live Salience", "🗨️ Multi-turn Dialogue Transcript", "📊 Batch File / Multi-Line Stream"],
        horizontal=True
    )
with ctl2:
    backend_sel = st.selectbox(
        "NLP Neural Engine Mode",
        ["🔀 Hybrid Ensembled Mode", "⚡ Ultra-Fast Lexical (<5ms)", "🧠 Deep RoBERTa Neural (PyTorch)"],
        index=0
    )
    if "Ultra-Fast" in backend_sel:
        clf.set_mode("lexical")
    elif "Deep RoBERTa" in backend_sel:
        clf.set_mode("transformer")
    else:
        clf.set_mode("hybrid")

st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

if mode == "📝 Single Message & Live Salience":
    st.markdown("### ⌖ Single Message Affect Decoder")

    # Integrated Live Microphone & Speech-to-Text Ingestion Console
    with st.expander("🎙️ Live Microphone & Spoken Voice Ingestion (Acoustic Prosody + Speech Recognition)", expanded=False):
        mic_c1, mic_c2 = st.columns([6, 6])
        with mic_c1:
            st.markdown("<div class='es-section-title'>🎙️ Browser Microphone Recorder</div>", unsafe_allow_html=True)
            voice_audio = st.audio_input("Record voice message", key="mic_audio_record")
        with mic_c2:
            st.markdown("<div class='es-section-title'>📁 Ingest Voice Note File</div>", unsafe_allow_html=True)
            uploaded_voice = st.file_uploader("Upload audio recording (WAV, MP3, OGG)", type=["wav", "mp3", "ogg"], key="upload_voice_record")

        active_audio_bytes = None
        if voice_audio is not None:
            active_audio_bytes = voice_audio.read()
        elif uploaded_voice is not None:
            active_audio_bytes = uploaded_voice.read()

        if active_audio_bytes:
            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
            st.audio(active_audio_bytes)

            with st.spinner("Analyzing vocal acoustic prosody & running speech-to-text..."):
                acoustics = prosody_ext.extract_from_file(active_audio_bytes)
                voice_res = voice_clf.classify_voice_emotion(acoustics)
                speech_res = transcriber.transcribe_audio_bytes(active_audio_bytes, voice_result=voice_res)

            # Display Voice Acoustic Indicators
            va1, va2, va3, va4 = st.columns(4)
            with va1:
                render_metric_card("Voice Dominant Affect", voice_res.dominant_emotion.upper(), delta=f"{int(voice_res.confidence * 100)}% Conf", color=EMOTION_COLORS.get(voice_res.dominant_emotion, "#3b82f6"))
            with va2:
                render_metric_card("Pitch F0", f"{acoustics.pitch_hz:.1f} Hz", delta="Fundamental Vocal F0", color="#10b981")
            with va3:
                render_metric_card("RMS Energy", f"{acoustics.rms_energy:.3f}", delta="Acoustic Loudness", color="#f59e0b")
            with va4:
                render_metric_card("Speech Stress", f"{int(voice_res.vocal_stress_level * 100)}%", delta=f"Jitter: {acoustics.jitter_percent:.1%}", color="#ef4444" if voice_res.vocal_stress_level > 0.5 else "#0ea5e9")

            if speech_res.full_transcript:
                st.success(f"🎙️ **Transcribed Speech:** \"{speech_res.full_transcript}\"")
                if st.button("⚡ Apply Transcribed Text to Message Decoder", key="btn_apply_transcript", use_container_width=True):
                    st.session_state.single_text_input = speech_res.full_transcript
                    st.rerun()

                # Display word tokens with phonetic prosody alignment
                if speech_res.tokens:
                    token_chips = " ".join([
                        f'<span style="display: inline-block; background: var(--surface-raised); border: 1px solid var(--border-color); border-radius: 4px; padding: 2px 8px; margin: 2px; font-size: 0.8rem; font-family: \'JetBrains Mono\', monospace;">'
                        f'<b>{tok.word}</b> <span style="color: #10b981; font-size: 0.72rem;">{tok.pitch_hz:.0f}Hz</span> '
                        f'<span style="color: var(--text-muted); font-size: 0.72rem;">[{tok.emotion_cue}]</span></span>'
                        for tok in speech_res.tokens
                    ])
                    st.markdown(f"<div style='margin-top: 6px;'><b>Phonetic Prosody Alignment:</b><br>{token_chips}</div>", unsafe_allow_html=True)
            else:
                st.info("💡 Vocal acoustics extracted. Speech-to-text returned no tokens (silence, offline, or background noise). You can type or paste the message in the text field below.")

    col_in, col_opts = st.columns([8, 4])
    with col_in:
        text_input = st.text_area(
            "Input Message Text",
            value=st.session_state.single_text_input,
            height=110,
            placeholder="Type or paste any message..."
        )
        st.session_state.single_text_input = text_input
    with col_opts:
        st.markdown("<div class='es-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='es-section-title'>⚡ Quick Presets</div>", unsafe_allow_html=True)
        if st.button("🎉 Joyful Milestone", use_container_width=True):
            st.session_state.single_text_input = "We finally hit 1 million users today! So proud of the entire engineering team! 🥳🍾"
            st.rerun()
        if st.button("😡 Outraged Customer", use_container_width=True):
            st.session_state.single_text_input = "This is the worst customer service ever. You stole my money and refuse to answer! Absolute scam!"
            st.rerun()
        if st.button("😰 Anxious / Stressed", use_container_width=True):
            st.session_state.single_text_input = "I'm terrified of failing the final exam tomorrow, having panic attacks and cannot sleep."
            st.rerun()
        if st.button("💙 Heartfelt Empathy", use_container_width=True):
            st.session_state.single_text_input = "I am so deeply sorry for your loss. Please know that we are all here to support you."
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


    if text_input.strip():
        res = clf.analyze_text(text_input)
        dom = res.dominant_emotion
        color = EMOTION_COLORS.get(dom, "#3b82f6")

        # Metric Tiles
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Dominant Affect", dom.upper(), delta=f"{int(res.confidence * 100)}% Conf", color=color)
        with m2:
            render_metric_card("Valence (Mood)", f"{res.affect.valence:+.2f}", delta="Negative ◄► Positive", color="#10b981" if res.affect.valence >= 0 else "#ef4444")
        with m3:
            render_metric_card("Arousal (Energy)", f"{res.affect.arousal:+.2f}", delta="Calm ◄► Excited", color="#f59e0b")
        with m4:
            render_metric_card("Dominance", f"{res.affect.dominance:+.2f}", delta="Submissive ◄► Empowered", color="#0ea5e9")

        st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

        # Visualizations
        v1, v2, v3 = st.columns([4, 4, 4])
        with v1:
            st.markdown("<div class='es-section-title'>Probability Distribution</div>", unsafe_allow_html=True)
            st.plotly_chart(render_emotion_horizontal_bars(res.probabilities), use_container_width=True)
        with v2:
            st.markdown("<div class='es-section-title'>8-Emotion Polar Radar</div>", unsafe_allow_html=True)
            st.plotly_chart(render_emotion_radar_chart(res.probabilities), use_container_width=True)
        with v3:
            st.markdown("<div class='es-section-title'>Russell's 2D VAD Circumplex</div>", unsafe_allow_html=True)
            st.plotly_chart(render_affect_quadrant_chart(res.affect), use_container_width=True)

        # Salience & Empathy
        e1, e2 = st.columns([6, 6])
        with e1:
            render_token_explainability_box(res.salience_tokens)
        with e2:
            render_empathy_advice_card(res.empathy_advice, dom)
            
            # Empathetic Reply Guidance
            st.markdown("""
            <div class="es-panel" style="margin-top: 0.65rem;">
                <div class="es-section-title">⚡ Empathetic Response Phrasing</div>
            """, unsafe_allow_html=True)
            
            if dom == "joy":
                replies = [
                    "That is fantastic news! Huge congratulations to you and the team! 🎉",
                    "I am so thrilled for you! Well deserved milestone! 🚀",
                    "Love seeing this win! Let's keep this amazing momentum going!"
                ]
            elif dom in ("anger", "disgust", "contempt"):
                replies = [
                    "I completely understand why you're frustrated. I am taking full ownership to resolve this right away.",
                    "I hear your concern loud and clear. Let's get on a call to fix this step-by-step.",
                    "Thank you for bringing this to my attention. I sincerely apologize for the inconvenience and will expedite a solution."
                ]
            elif dom in ("sadness", "fear"):
                replies = [
                    "I am really sorry you're going through this. Please know I am here if you need anything at all.",
                    "Take all the time you need. We have your back and will support you through this.",
                    "Sending you warm thoughts and strength. Don't hesitate to reach out whenever you're ready."
                ]
            else:
                replies = [
                    "Got it, thanks for the update! I will proceed accordingly.",
                    "Thank you for sharing this information. Let me know if you need any further details.",
                    "Understood. I will keep you posted on our progress."
                ]

            for r in replies:
                st.markdown(f'<div style="background: var(--surface-raised); border: 1px solid var(--border-color); padding: 6px 10px; border-radius: 4px; margin-bottom: 5px; font-size: 0.82rem; color: #e2e8f0; border-left: 2px solid #3b82f6; font-family: \'JetBrains Mono\', monospace;">💬 {r}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


elif mode == "🗨️ Multi-turn Dialogue Transcript":
    st.markdown("### ⌖ Conversational Flow & Escalation Studio")

    default_transcript = """[10:15 AM] User: Hello, I have an urgent issue with my cloud deployment.
[10:16 AM] Agent: Hi! I am here to help you. What error code are you encountering?
[10:17 AM] User: It says 500 Internal Server Error and all our customer checkout lines are dead!! I am losing revenue every minute!!
[10:18 AM] Agent: I understand how critical this is. I am escalating directly to our Tier 3 infrastructure engineers right now.
[10:19 AM] User: Please hurry, this is terrifying for our business.
[10:21 AM] Agent: The server route was unblocked and traffic is fully normal now. We have applied a credit to your account.
[10:22 AM] User: Wow, thank goodness! It's working perfectly now. Thank you so much for the rapid resolution! 🎉"""

    transcript_in = st.text_area("Paste Chat Transcript", value=default_transcript, height=160)

    if transcript_in.strip():
        summary = conv.parse_and_analyze_transcript(transcript_in)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_metric_card("Total Turns", str(summary.total_turns), delta=f"{len(summary.speakers)} Speakers", color="#3b82f6")
        with k2:
            esc_col = "#10b981" if summary.escalation_risk == "Low" else ("#f59e0b" if summary.escalation_risk == "Moderate" else "#ef4444")
            render_metric_card("Escalation Risk", summary.escalation_risk, delta="Conflict Sentinel", color=esc_col)
        with k3:
            render_metric_card("Empathy / Rapport", f"{int(summary.rapport_empathy_score * 100)}%", delta="Mirroring Synchrony", color="#0ea5e9")
        with k4:
            render_metric_card("Turning Points", str(len(summary.turning_points)), delta="Inflection Shifts", color="#f59e0b")

        st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

        col_chat, col_flow = st.columns([5, 6])
        with col_chat:
            st.markdown("<div class='es-section-title'>Message Stream Diagnostic</div>", unsafe_allow_html=True)
            for turn in summary.turns:
                is_right = (turn.speaker == summary.speakers[-1]) if len(summary.speakers) > 1 else False
                render_chat_bubble(turn, is_right=is_right)

        with col_flow:
            st.markdown("<div class='es-section-title'>Conversational Mood & Energy Trajectory</div>", unsafe_allow_html=True)
            st.plotly_chart(render_conversation_flow_chart(summary.emotional_trajectory), use_container_width=True)

            if summary.turning_points:
                st.markdown("<div class='es-section-title'>⚡ Turning Points Detected</div>", unsafe_allow_html=True)
                for tp in summary.turning_points:
                    st.markdown(f"""
                    <div style="background: var(--surface-card); border: 1px solid var(--border-color); border-left: 3px solid #f59e0b; padding: 6px 12px; border-radius: 4px; margin-bottom: 6px; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;">
                        <b>Turn #{tp['turn_index']} [{tp['speaker']}]</b>: <code>{tp['from_emotion']}</code> ➔ <code style="color: #f59e0b;">{tp['to_emotion']}</code> (ΔValence: {tp['valence_delta']:+.2f})
                    </div>
                    """, unsafe_allow_html=True)


elif mode == "📊 Batch File / Multi-Line Stream":
    st.markdown("### ⌖ Batch Text Affect Intelligence & File Ingestion")

    up_file = st.file_uploader("Upload CSV or TXT File", type=["csv", "txt"])
    
    batch_text = ""
    if up_file is not None:
        if up_file.name.endswith(".csv"):
            raw_df = pd.read_csv(up_file)
            str_cols = [c for c in raw_df.columns if raw_df[c].dtype == "object"]
            if str_cols:
                batch_text = "\n".join(raw_df[str_cols[0]].dropna().astype(str).tolist())
            else:
                batch_text = "\n".join(raw_df.iloc[:, 0].dropna().astype(str).tolist())
        else:
            batch_text = up_file.read().decode("utf-8")
    else:
        batch_text = st.text_area(
            "Or Paste Lines of Text (One Message Per Line)",
            value="""I am so thrilled with this achievement! 🎉
The package was damaged and customer service was unresponsive. 😡
Regular update, nothing remarkable.
Anxious about the upcoming review tomorrow...
Thank you for being so supportive and kind. ❤️
Utterly disgusting experience at the restaurant. 🤮
Astonishing performance, way beyond expectations! 🚀""",
            height=140
        )

    if batch_text.strip():
        lines = [line.strip() for line in batch_text.split("\n") if line.strip()]
        results, df = conv.analyze_batch_messages(lines)

        st.markdown(f"<div class='es-section-title'>Analyzed {len(results)} distinct items</div>", unsafe_allow_html=True)
        
        bc1, bc2 = st.columns([5, 7])
        with bc1:
            st.markdown("<div class='es-section-title'>Batch Affect Distribution</div>", unsafe_allow_html=True)
            counts = df["Dominant Emotion"].value_counts().to_dict()
            st.plotly_chart(render_emotion_distribution_pie(counts), use_container_width=True)

        with bc2:
            st.markdown("<div class='es-section-title'>Telemetry Data Table</div>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=250)

        # Export Buttons
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "📥 Export Analyzed Batch to CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="batch_text_affect_report.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d2:
            st.download_button(
                "📥 Export Analyzed Batch to JSON",
                data=df.to_json(orient="records", indent=2),
                file_name="batch_text_affect_report.json",
                mime="application/json",
                use_container_width=True
            )

# Real-Time WebSocket Streaming Microservice Telemetry
st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
with st.expander("📡 Real-Time Microservice Streaming Telemetry (`/ws/stream-speech` & `/ws/stream-affect`)", expanded=False):
    st.markdown("""
    <div style="font-size: 0.85rem; color: var(--text-sub); margin-bottom: 0.8rem;">
        EmotionSense exposes bi-directional streaming WebSocket gateways for sub-10ms streaming text and live speech transcription decoding.
    </div>
    """, unsafe_allow_html=True)

    ws_col1, ws_col2 = st.columns([6, 6])
    with ws_col1:
        st.markdown("<div class='es-section-title'>WebSocket Gateway Endpoints</div>", unsafe_allow_html=True)
        st.code("ws://127.0.0.1:8000/ws/stream-affect\nws://127.0.0.1:8000/ws/stream-speech", language="text")
        st.caption("Protocol: Bi-directional JSON stream over WebSocket frame protocol.")

    with ws_col2:
        st.markdown("<div class='es-section-title'>Simulated Gateway Tester</div>", unsafe_allow_html=True)
        ws_test_phrase = st.text_input("Stream phrase to simulate", value="I am completely amazed by how responsive this affective pipeline is! ⚡")
        if st.button("Simulate WebSocket Payload Stream", use_container_width=True):
            speech_stream_out = transcriber.transcribe_text_stream(ws_test_phrase)
            st.json({
                "endpoint": "/ws/stream-speech",
                "transcript": speech_stream_out.full_transcript,
                "dominant_emotion": speech_stream_out.text_emotion.dominant_emotion if speech_stream_out.text_emotion else "neutral",
                "confidence": speech_stream_out.text_emotion.confidence if speech_stream_out.text_emotion else 0.0,
                "valence": speech_stream_out.text_emotion.affect.valence if speech_stream_out.text_emotion else 0.0,
                "tokens_count": len(speech_stream_out.tokens),
                "timestamp": speech_stream_out.timestamp,
            })

