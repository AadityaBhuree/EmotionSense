"""EmotionSense — Enterprise Multimodal Affective Intelligence & Real-Time Emotion Recognition Platform."""

import streamlit as st
import time
import pandas as pd
from typing import Dict, List, Any

# Page Configuration
st.set_page_config(
    page_title="EmotionSense | Affective Intelligence Workstation",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import APP_NAME, APP_TAGLINE, APP_ICON, THEME_COLORS, EMOTION_COLORS
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

# Apply precision neuro-instrument styling
inject_modern_styles()

# Initialize text classifiers in session state
if "hybrid_classifier" not in st.session_state:
    st.session_state.hybrid_classifier = HybridEmotionClassifier(mode="hybrid")
if "conversation_analyzer" not in st.session_state:
    st.session_state.conversation_analyzer = ConversationAffectAnalyzer(st.session_state.hybrid_classifier)

clf = st.session_state.hybrid_classifier
conv_analyzer: ConversationAffectAnalyzer = st.session_state.conversation_analyzer

# Sidebar Navigation & Telemetry
with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.25rem;">
        <span style="font-size: 2rem;">{APP_ICON}</span>
        <div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">{APP_NAME}</div>
            <div style="font-size: 0.72rem; color: #3b82f6; font-weight: 600; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">Affective AI Console</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 NLP Engine Mode")
    selected_backend = st.selectbox(
        "Active NLP Engine",
        ["🔀 Hybrid Ensembled Mode", "⚡ Ultra-Fast Lexical (<5ms)", "🧠 Deep RoBERTa Neural (PyTorch)"],
        index=0
    )
    if "Ultra-Fast" in selected_backend:
        clf.set_mode("lexical")
    elif "Deep RoBERTa" in selected_backend:
        clf.set_mode("transformer")
    else:
        clf.set_mode("hybrid")

    st.markdown("---")
    st.markdown("### ⚡ Multimodal Engine Status")
    st.markdown(f"""
    - **NLP Engine**: `{clf.mode.upper()}` <span class="es-pill es-pill-active" style="padding:1px 5px; font-size:0.65rem;">ACTIVE</span>
    - **Dialogue Flow**: `Trajectory + Escalation`
    - **Vision Engine**: `MediaPipe 468 Mesh`
    - **Audio Engine**: `Acoustic Prosody F0`
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Workstation Tip")
    st.caption("Paste any text message, customer chat transcript, or multi-line batch file to trigger real-time affective prediction.")

# Main Workstation Console Header
render_header("EmotionSense Workstation", "Multimodal Affective Intelligence & Real-Time Emotion Decoder")

# Key Platform Telemetry Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Affect Taxonomy", "8 Core + 8 Nuance", delta="Plutchik/Ekman", color="#3b82f6", subtext="Calibrated Spectrum")
with c2:
    render_metric_card("Coordinate Space", "3D VAD", delta="Russell Circumplex", color="#0ea5e9", subtext="Valence • Arousal • Dominance")
with c3:
    render_metric_card("Dialogue Dynamics", "Multi-Turn", delta="Escalation Sentinel", color="#10b981", subtext="Trajectory Tracking")
with c4:
    render_metric_card("Inference Latency", "< 4.8 ms", delta="Zero-Lag Local", color="#f59e0b", subtext="Hardware Optimized")

st.markdown("<div style='margin-bottom: 0.85rem;'></div>", unsafe_allow_html=True)

# Main Studio Tabs
tab1, tab2, tab3 = st.tabs([
    "📝 Single Text Message Analyzer",
    "🗨️ Multi-Turn Dialogue & Escalation",
    "📊 Batch Text Stream & Dataset"
])

# -------------------------------------------------------------
# TAB 1: Single Text Message
# -------------------------------------------------------------
with tab1:
    st.markdown("#### ⌖ Text Input & Instant Affect Scope")

    # Preset selector styled as tactile instrument keys
    st.markdown("<div class='es-section-title'>⚡ Scenario Presets</div>", unsafe_allow_html=True)
    preset_cols = st.columns([1, 1, 1, 1, 1])
    with preset_cols[0]:
        if st.button("🎉 Milestone Win", use_container_width=True):
            st.session_state.single_input = "I just got promoted to Lead Architect and the whole team is celebrating!! 🎉🥳"
    with preset_cols[1]:
        if st.button("🤬 Service Outrage", use_container_width=True):
            st.session_state.single_input = "I am utterly disgusted by your broken service. This is completely unacceptable and ruined my day!"
    with preset_cols[2]:
        if st.button("😰 Severe Anxiety", use_container_width=True):
            st.session_state.single_input = "I have a terrible feeling about this presentation tomorrow, I'm shaking and panicked."
    with preset_cols[3]:
        if st.button("💙 Deep Gratitude", use_container_width=True):
            st.session_state.single_input = "Thank you from the bottom of my heart for being there for me during this tough time."
    with preset_cols[4]:
        if st.button("😒 Subtle Sarcasm", use_container_width=True):
            st.session_state.single_input = "Oh wonderful, another meeting that could have been an email. Truly thrilling. 🙄"

    if "single_input" not in st.session_state:
        st.session_state.single_input = "I am absolutely thrilled and overjoyed to share our latest breakthroughs today! 🚀✨"

    user_text = st.text_area(
        "Message Text for Affective Decoding",
        value=st.session_state.single_input,
        height=95,
        placeholder="Paste any SMS, tweet, email snippet, or customer message here...",
        key="text_single_input"
    )

    char_count = len(user_text)
    word_count = len(user_text.split()) if user_text else 0
    st.markdown(f"""
    <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #526079; margin-top: -8px; margin-bottom: 12px;">
        LENGTH: {char_count} chars | {word_count} words
    </div>
    """, unsafe_allow_html=True)

    if user_text.strip():
        res = clf.analyze_text(user_text)
        dom_emo = res.dominant_emotion
        color = EMOTION_COLORS.get(dom_emo, "#3b82f6")

        # Instrument Readout Bar
        st.markdown(f"""
        <div class="es-panel" style="border-left: 4px solid {color}; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background: {color}; box-shadow: 0 0 8px {color};"></div>
                    <div>
                        <div style="font-size: 0.7rem; color: var(--text-sub); text-transform: uppercase; letter-spacing: 0.06em; font-family: 'JetBrains Mono', monospace;">
                            Dominant Affect Detection
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
                            {dom_emo.capitalize()}
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 24px; font-family: 'JetBrains Mono', monospace;">
                    <div style="text-align: right;">
                        <div style="font-size: 0.68rem; color: var(--text-sub); text-transform: uppercase; letter-spacing: 0.05em;">Confidence</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: {color};">{int(res.confidence * 100)}%</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.68rem; color: var(--text-sub); text-transform: uppercase; letter-spacing: 0.05em;">Valence (Mood)</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: {'#10b981' if res.affect.valence >= 0 else '#ef4444'};">{res.affect.valence:+.2f}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.68rem; color: var(--text-sub); text-transform: uppercase; letter-spacing: 0.05em;">Arousal (Energy)</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #f59e0b;">{res.affect.arousal:+.2f}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts Row: Probabilities Bar Chart, 8-Emotion Radar, 2D Circumplex Quadrant
        col_c1, col_c2, col_c3 = st.columns([4, 4, 4])
        with col_c1:
            st.markdown("<div class='es-section-title'>Ranked Probabilities</div>", unsafe_allow_html=True)
            st.plotly_chart(render_emotion_horizontal_bars(res.probabilities), use_container_width=True)
        with col_c2:
            st.markdown("<div class='es-section-title'>8-Emotion Polar Radar</div>", unsafe_allow_html=True)
            st.plotly_chart(render_emotion_radar_chart(res.probabilities), use_container_width=True)
        with col_c3:
            st.markdown("<div class='es-section-title'>Russell's Circumplex (2D VAD)</div>", unsafe_allow_html=True)
            st.plotly_chart(render_affect_quadrant_chart(res.affect), use_container_width=True)

        # Explainability & Recommendations
        exp_col1, exp_col2 = st.columns([6, 6])
        with exp_col1:
            render_token_explainability_box(res.salience_tokens)
        with exp_col2:
            render_empathy_advice_card(res.empathy_advice, dom_emo)
            if res.nuanced_emotions:
                nuance_badges = " ".join([
                    f'<span class="es-pill" style="background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); font-size: 0.7rem;">{k.capitalize()}: {int(v*100)}%</span>'
                    for k, v in res.nuanced_emotions.items()
                ])
                st.markdown(f"""
                <div style="margin-top: 0.65rem; padding: 0.65rem 0.85rem; background: var(--surface-card); border-radius: 6px; border: 1px solid var(--border-color);">
                    <div style="font-size: 0.7rem; color: var(--text-sub); text-transform: uppercase; font-family: 'JetBrains Mono', monospace; font-weight: 600; margin-bottom: 5px;">
                        Sub-Categorical Nuance Spectrum
                    </div>
                    <div>{nuance_badges}</div>
                </div>
                """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: Multi-Message & Dialogue Thread
# -------------------------------------------------------------
with tab2:
    st.markdown("#### ⌖ Multi-Turn Dialogue Transcript & Escalation Sentinel")

    # Dialogue Presets
    st.markdown("<div class='es-section-title'>⚡ Dialogue Scenarios</div>", unsafe_allow_html=True)
    d_preset_cols = st.columns([1, 1, 1])
    with d_preset_cols[0]:
        if st.button("🚨 Service Escalation & Recovery", use_container_width=True):
            st.session_state.dialogue_input = """[10:00 AM] Customer: I have been waiting for 4 hours and nothing is working. This is totally unacceptable!!
[10:01 AM] Agent: I am terribly sorry for the delay and frustration. Let me immediately check your order priority.
[10:02 AM] Customer: You better fix it, I lost an important client today because of this.
[10:03 AM] Agent: I completely understand your anger. I have just refunded your fees and expedited the delivery with priority dispatch.
[10:04 AM] Customer: Wow, thank you for resolving it so quickly. I really appreciate your help. ❤️"""
    with d_preset_cols[1]:
        if st.button("💬 Peer Empathy & Burnout Support", use_container_width=True):
            st.session_state.dialogue_input = """Alex: Hey, are you okay? You seemed really quiet in today's sync.
Sam: Honestly, I'm feeling so burnt out and anxious about the upcoming deadline.
Alex: I hear you. You've been carrying so much load lately. Take a breather, I'll take over the frontend tasks today.
Sam: That means the world to me, thank you so much for being such an awesome friend! 😊"""
    with d_preset_cols[2]:
        if st.button("📈 Product Feedback & Delight", use_container_width=True):
            st.session_state.dialogue_input = """Reviewer: The device looks great, but the battery life is quite disappointing.
Support: Thanks for the feedback! Have you installed the latest v2.4 firmware update which fixes power drain?
Reviewer: Just installed it now. Oh wow, battery is lasting twice as long! Extremely impressed! 🎉"""

    if "dialogue_input" not in st.session_state:
        st.session_state.dialogue_input = """[09:15 AM] Alex: Good morning! Are we ready for the big client demo today?
[09:16 AM] Jordan: I'm a bit nervous, but all our test pipelines are green and looking sharp!
[09:17 AM] Alex: Awesome, let's crush this presentation! 🚀
[09:18 AM] Jordan: Absolutely, thrilled to show what we built! 🎉"""

    dialogue_text = st.text_area(
        "Conversation Transcript (format: 'Speaker: text' or '[Time] Speaker: text')",
        value=st.session_state.dialogue_input,
        height=130,
        key="text_dialogue_input"
    )

    if dialogue_text.strip():
        summary = conv_analyzer.parse_and_analyze_transcript(dialogue_text)

        # Dialogue Summary KPI Bar
        dkpi1, dkpi2, dkpi3, dkpi4 = st.columns(4)
        with dkpi1:
            render_metric_card("Total Turns", str(summary.total_turns), delta=f"{len(summary.speakers)} Speakers", color="#3b82f6")
        with dkpi2:
            esc_color = "#10b981" if summary.escalation_risk == "Low" else ("#f59e0b" if summary.escalation_risk == "Moderate" else "#ef4444")
            render_metric_card("Escalation Risk", summary.escalation_risk, delta="Conflict Sentinel", color=esc_color)
        with dkpi3:
            render_metric_card("Empathy / Rapport", f"{int(summary.rapport_empathy_score * 100)}%", delta="Affective Synchrony", color="#0ea5e9")
        with dkpi4:
            render_metric_card("Turning Points", str(len(summary.turning_points)), delta="Shift Moments", color="#f59e0b")

        st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

        # Two Columns: Left = Interactive Chat Timeline, Right = Flow Graph & Turning Points
        d_left, d_right = st.columns([5, 6])
        with d_left:
            st.markdown("<div class='es-section-title'>Turn-by-Turn Diagnostic Stream</div>", unsafe_allow_html=True)
            for turn in summary.turns:
                is_right_align = (turn.speaker == summary.speakers[-1]) if len(summary.speakers) > 1 else False
                render_chat_bubble(turn, is_right=is_right_align)

        with d_right:
            st.markdown("<div class='es-section-title'>Conversational Mood & Energy Trajectory</div>", unsafe_allow_html=True)
            st.plotly_chart(render_conversation_flow_chart(summary.emotional_trajectory), use_container_width=True)

            if summary.turning_points:
                st.markdown("<div class='es-section-title'>⚡ Detected Emotional Turning Points</div>", unsafe_allow_html=True)
                for tp in summary.turning_points:
                    st.markdown(f"""
                    <div style="background: var(--surface-card); border: 1px solid var(--border-color); border-left: 3px solid #f59e0b; padding: 6px 12px; border-radius: 4px; margin-bottom: 6px; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;">
                        <b>Turn #{tp['turn_index']} [{tp['speaker']}]</b>: Shifted <code>{tp['from_emotion']}</code> ➔ <code style="color: #f59e0b;">{tp['to_emotion']}</code> (ΔValence: {tp['valence_delta']:+.2f})
                    </div>
                    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: Batch Text Stream & Dataset
# -------------------------------------------------------------
with tab3:
    st.markdown("#### ⌖ Batch Message Stream & Dataset Telemetry")
    
    batch_raw = st.text_area(
        "Batch Messages (one message per line)",
        value="""I love this product so much, it changed my workflow! ❤️
The packaging arrived completely crushed and unusable. Worst experience ever.
The delivery was on time, standard service.
I am so anxious about my exam results tomorrow morning...
Unbelievable quality, absolutely stunning design! 🎉
This app keeps crashing every time I try to save. Very annoying.
Thank you for your quick and kind assistance! 🙏""",
        height=130,
        key="text_batch_input"
    )

    if batch_raw.strip():
        lines = [line.strip() for line in batch_raw.split("\n") if line.strip()]
        results, df = conv_analyzer.analyze_batch_messages(lines)

        st.markdown(f"<div class='es-section-title'>Processed {len(results)} Messages</div>", unsafe_allow_html=True)
        
        # Batch Visualization Row
        b_col1, b_col2 = st.columns([5, 7])
        with b_col1:
            st.markdown("<div class='es-section-title'>Batch Affect Distribution</div>", unsafe_allow_html=True)
            emo_counts = df["Dominant Emotion"].value_counts().to_dict()
            st.plotly_chart(render_emotion_distribution_pie(emo_counts), use_container_width=True)

        with b_col2:
            st.markdown("<div class='es-section-title'>Dataset Inspection Table</div>", unsafe_allow_html=True)
            st.dataframe(df[["Index", "Dominant Emotion", "Confidence (%)", "Valence", "Polarity", "Message"]], use_container_width=True, height=240)

        # Download Buttons
        dl1, dl2 = st.columns([1, 1])
        with dl1:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Batch Telemetry (CSV)",
                data=csv_data,
                file_name="emotionsense_batch_affect.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl2:
            json_data = df.to_json(orient="records", indent=2)
            st.download_button(
                "📥 Export Batch Telemetry (JSON)",
                data=json_data,
                file_name="emotionsense_batch_affect.json",
                mime="application/json",
                use_container_width=True
            )

st.markdown("---")

# Quick Navigation to Other Studios
st.markdown("### 🛠️ Specialized Workstations")
mod1, mod2, mod3 = st.columns(3)
with mod1:
    st.markdown("""
    <div class="es-panel" style="height: 100%;">
        <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">🎥</div>
        <div style="font-size: 1rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.25rem;">Live Vision & WebRTC Studio</div>
        <p style="color: var(--text-sub); font-size: 0.82rem; line-height: 1.45; margin: 0;">
            Track 468-point 3D facial mesh, micro-expressions, head pose, and FACS action units in real time.
        </p>
    </div>
    """, unsafe_allow_html=True)

with mod2:
    st.markdown("""
    <div class="es-panel" style="height: 100%;">
        <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">📁</div>
        <div style="font-size: 1rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.25rem;">Video & Audio File Analysis</div>
        <p style="color: var(--text-sub); font-size: 0.82rem; line-height: 1.45; margin: 0;">
            Inspect pre-recorded MP4, WAV, and MP3 files with acoustic prosody and frame-by-frame affective timelines.
        </p>
    </div>
    """, unsafe_allow_html=True)

with mod3:
    st.markdown("""
    <div class="es-panel" style="height: 100%;">
        <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">📑</div>
        <div style="font-size: 1rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.25rem;">Session History & Reports</div>
        <p style="color: var(--text-sub); font-size: 0.82rem; line-height: 1.45; margin: 0;">
            Scrub past recording sessions, review emotional turning points, and export clinical diagnostic reports.
        </p>
    </div>
    """, unsafe_allow_html=True)
