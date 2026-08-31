"""EmotionSense — Enterprise Multimodal Affective Intelligence & Real-Time Emotion Recognition Platform."""

import streamlit as st
import time
import pandas as pd
from typing import Dict, List, Any

# Page Configuration
st.set_page_config(
    page_title="EmotionSense | Affective AI Platform",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import APP_NAME, APP_TAGLINE, APP_ICON, THEME_COLORS, EMOTION_COLORS
from src.ui.styles import inject_glassmorphic_styles
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
from src.text import TextEmotionClassifier, ConversationAffectAnalyzer

# Apply custom dark glassmorphic styling
inject_glassmorphic_styles()

# Initialize text classifiers in session state
if "text_classifier" not in st.session_state:
    st.session_state.text_classifier = TextEmotionClassifier()
if "conversation_analyzer" not in st.session_state:
    st.session_state.conversation_analyzer = ConversationAffectAnalyzer(st.session_state.text_classifier)

clf: TextEmotionClassifier = st.session_state.text_classifier
conv_analyzer: ConversationAffectAnalyzer = st.session_state.conversation_analyzer

# Sidebar Navigation & Telemetry
with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem;">
        <span style="font-size: 2.2rem;">{APP_ICON}</span>
        <div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">{APP_NAME}</div>
            <div style="font-size: 0.75rem; color: #818cf8; font-weight: 600; text-transform: uppercase;">Affective AI v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚡ Multimodal Engine Status")
    st.markdown("""
    - **Text / NLP Engine**: `Active (VAD + 8-Ekman + Salience)` ✅
    - **Dialogue Flow Engine**: `Active (Trajectory + Escalation)` ✅
    - **Vision Engine**: `MediaPipe 468 FaceMesh` ✅
    - **Audio Engine**: `Acoustic Prosody (F0/Jitter)` ✅
    """)

    st.markdown("---")
    st.markdown("### 💡 Quick Tip")
    st.caption("Paste any single text message, customer chat log, or multiple message lines to immediately trigger real-time affective predictions.")

# Main Landing Header
render_header("EmotionSense", "Multimodal Affective Intelligence & Real-Time Emotion Decoder")

# Hero Banner
st.markdown("""
<div class="es-card" style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.45) 100%);">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
        <div style="max-width: 750px;">
            <h2 style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem;">
                Decode Emotions from Text, Voice & Micro-Expressions
            </h2>
            <p style="color: #94a3b8; font-size: 1rem; line-height: 1.6;">
                Instantly decode emotional states, sentiment polarity, and continuous 3D Valence-Arousal-Dominance (VAD) spaces from single text messages, multi-turn chat transcripts, or batch streams.
            </p>
        </div>
        <div style="display: flex; gap: 10px;">
            <div class="es-pill es-pill-active" style="padding: 6px 14px; font-size: 0.85rem;">⚡ Real-time Inference</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Key Platform Telemetry Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Taxonomy", "8 Core + 8 Nuances", delta="Ekman & Plutchik", color="#6366f1")
with c2:
    render_metric_card("Affect Space", "3D VAD", delta="Valence • Arousal • Dominance", color="#06b6d4")
with c3:
    render_metric_card("Dialogue Flow", "Multi-Turn", delta="Trajectory & Escalation", color="#10b981")
with c4:
    render_metric_card("Inference Latency", "< 5 ms", delta="Zero-Lag Local NLP", color="#f59e0b")

st.markdown("<br>", unsafe_allow_html=True)

# Main Studio Tabs
st.markdown("### 💬 Live Emotion Prediction Studio")
tab1, tab2, tab3 = st.tabs([
    "📝 Single Text Message",
    "🗨️ Multi-Message & Dialogue Thread",
    "📊 Batch Text Stream & Dataset"
])

# -------------------------------------------------------------
# TAB 1: Single Text Message
# -------------------------------------------------------------
with tab1:
    st.markdown("#### Paste or type a text message for instant affective prediction:")

    # Preset selector
    preset_cols = st.columns([1, 1, 1, 1, 1])
    preset_text = ""
    with preset_cols[0]:
        if st.button("🎉 Celebration", use_container_width=True):
            st.session_state.single_input = "I just got promoted to Lead Architect and the whole team is celebrating!! 🎉🥳"
    with preset_cols[1]:
        if st.button("🤬 Frustration", use_container_width=True):
            st.session_state.single_input = "I am utterly disgusted by your broken service. This is completely unacceptable and ruined my day!"
    with preset_cols[2]:
        if st.button("😰 Anxiety", use_container_width=True):
            st.session_state.single_input = "I have a terrible feeling about this presentation tomorrow, I'm shaking and panicked."
    with preset_cols[3]:
        if st.button("💙 Gratitude", use_container_width=True):
            st.session_state.single_input = "Thank you from the bottom of my heart for being there for me during this tough time."
    with preset_cols[4]:
        if st.button("😒 Passive-Aggressive", use_container_width=True):
            st.session_state.single_input = "Oh wonderful, another meeting that could have been an email. Truly thrilling. 🙄"

    if "single_input" not in st.session_state:
        st.session_state.single_input = "I am absolutely thrilled and overjoyed to share our latest breakthroughs today! 🚀✨"

    user_text = st.text_area(
        "Message Text",
        value=st.session_state.single_input,
        height=100,
        placeholder="Paste any SMS, tweet, email snippet, or customer message here...",
        key="text_single_input"
    )

    if user_text.strip():
        res = clf.analyze_text(user_text)
        dom_emo = res.dominant_emotion
        color = EMOTION_COLORS.get(dom_emo, "#6366f1")

        # Top Result Summary Card
        st.markdown(f"""
        <div style="
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid {color}66;
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 0 25px {color}22;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="width: 16px; height: 16px; border-radius: 50%; background: {color}; box-shadow: 0 0 12px {color};"></div>
                    <div>
                        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Dominant Affect</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #f8fafc;">{dom_emo.capitalize()}</div>
                    </div>
                </div>
                <div style="display: flex; gap: 24px; font-family: 'JetBrains Mono', monospace;">
                    <div style="text-align: right;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Confidence</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{int(res.confidence * 100)}%</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Sentiment Polarity</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {'#10b981' if res.polarity >= 0 else '#ef4444'};">{res.polarity:+.2f}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts Row: Probabilities Bar Chart, 8-Emotion Radar, 2D Circumplex Quadrant
        col_c1, col_c2, col_c3 = st.columns([4, 4, 4])
        with col_c1:
            st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8; font-weight:600;'>Top Emotion Probabilities</p>", unsafe_allow_html=True)
            st.plotly_chart(render_emotion_horizontal_bars(res.probabilities), use_container_width=True)
        with col_c2:
            st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8; font-weight:600;'>8-Emotion Polar Radar</p>", unsafe_allow_html=True)
            st.plotly_chart(render_emotion_radar_chart(res.probabilities), use_container_width=True)
        with col_c3:
            st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8; font-weight:600;'>Russell's 2D VAD Circumplex</p>", unsafe_allow_html=True)
            st.plotly_chart(render_affect_quadrant_chart(res.affect), use_container_width=True)

        # Explainability & Recommendations
        exp_col1, exp_col2 = st.columns([6, 6])
        with exp_col1:
            render_token_explainability_box(res.salience_tokens)
        with exp_col2:
            render_empathy_advice_card(res.empathy_advice, dom_emo)
            if res.nuanced_emotions:
                st.markdown("""
                <div style="margin-top: 0.75rem; padding: 0.75rem 1rem; background: rgba(15, 23, 42, 0.5); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 6px;">Detected Nuanced Affects</div>
                """, unsafe_allow_html=True)
                nuance_badges = " ".join([f'<span class="es-pill" style="background: rgba(99,102,241,0.2); color: #818cf8; border: 1px solid rgba(99,102,241,0.4);">{k.capitalize()}: {int(v*100)}%</span>' for k, v in res.nuanced_emotions.items()])
                st.markdown(f"<div>{nuance_badges}</div></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: Multi-Message & Dialogue Thread
# -------------------------------------------------------------
with tab2:
    st.markdown("#### Paste a multi-turn conversation or chat transcript:")

    # Dialogue Presets
    d_preset_cols = st.columns([1, 1, 1])
    with d_preset_cols[0]:
        if st.button("🚨 Support Escalation & Resolution", use_container_width=True):
            st.session_state.dialogue_input = """[10:00 AM] Customer: I have been waiting for 4 hours and nothing is working. This is totally unacceptable!!
[10:01 AM] Agent: I am terribly sorry for the delay and frustration. Let me immediately check your order priority.
[10:02 AM] Customer: You better fix it, I lost an important client today because of this.
[10:03 AM] Agent: I completely understand your anger. I have just refunded your fees and expedited the delivery with priority dispatch.
[10:04 AM] Customer: Wow, thank you for resolving it so quickly. I really appreciate your help. ❤️"""
    with d_preset_cols[1]:
        if st.button("💬 Friendly Empathy Chat", use_container_width=True):
            st.session_state.dialogue_input = """Alex: Hey, are you okay? You seemed really quiet in today's sync.
Sam: Honestly, I'm feeling so burnt out and anxious about the upcoming deadline.
Alex: I hear you. You've been carrying so much load lately. Take a breather, I'll take over the frontend tasks today.
Sam: That means the world to me, thank you so much for being such an awesome friend! 😊"""
    with d_preset_cols[2]:
        if st.button("📈 Product Review Dialogue", use_container_width=True):
            st.session_state.dialogue_input = """Reviewer: The device looks great, but the battery life is quite disappointing.
Support: Thanks for the feedback! Have you installed the latest v2.4 firmware update which fixes power drain?
Reviewer: Just installed it now. Oh wow, battery is lasting twice as long! Extremely impressed! 🎉"""

    if "dialogue_input" not in st.session_state:
        st.session_state.dialogue_input = """[09:15 AM] Alex: Good morning! Are we ready for the big client demo today?
[09:16 AM] Jordan: I'm a bit nervous, but all our test pipelines are green and looking sharp!
[09:17 AM] Alex: Awesome, let's crush this presentation! 🚀
[09:18 AM] Jordan: Absolutely, thrilled to show what we built! 🎉"""

    dialogue_text = st.text_area(
        "Conversation Transcript (Supports 'Speaker: text' or '[Time] Speaker: text')",
        value=st.session_state.dialogue_input,
        height=140,
        key="text_dialogue_input"
    )

    if dialogue_text.strip():
        summary = conv_analyzer.parse_and_analyze_transcript(dialogue_text)

        # Dialogue Summary KPI Bar
        dkpi1, dkpi2, dkpi3, dkpi4 = st.columns(4)
        with dkpi1:
            render_metric_card("Total Messages", str(summary.total_turns), delta=f"{len(summary.speakers)} Participants", color="#6366f1")
        with dkpi2:
            esc_color = "#10b981" if summary.escalation_risk == "Low" else ("#f59e0b" if summary.escalation_risk == "Moderate" else "#ef4444")
            render_metric_card("Escalation Risk", summary.escalation_risk, delta="Conflict Monitoring", color=esc_color)
        with dkpi3:
            render_metric_card("Empathy / Rapport", f"{int(summary.rapport_empathy_score * 100)}%", delta="Affective Synchrony", color="#06b6d4")
        with dkpi4:
            render_metric_card("Turning Points", str(len(summary.turning_points)), delta="Inflection Moments", color="#f59e0b")

        # Two Columns: Left = Interactive Chat Timeline, Right = Flow Graph & Turning Points
        d_left, d_right = st.columns([5, 6])
        with d_left:
            st.markdown("#### 💬 Message Emotion Breakdown")
            for turn in summary.turns:
                is_right_align = (turn.speaker == summary.speakers[-1]) if len(summary.speakers) > 1 else False
                render_chat_bubble(turn, is_right=is_right_align)

        with d_right:
            st.markdown("#### 📈 Conversational Mood & Energy Flow")
            st.plotly_chart(render_conversation_flow_chart(summary.emotional_trajectory), use_container_width=True)

            if summary.turning_points:
                st.markdown("##### ⚡ Key Emotional Turning Points")
                for tp in summary.turning_points:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.45); border-left: 3px solid #f59e0b; padding: 6px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 0.85rem;">
                        <b>Turn #{tp['turn_index']} ({tp['speaker']})</b>: Shifted from <code>{tp['from_emotion']}</code> to <code style="color: #f59e0b;">{tp['to_emotion']}</code> (ΔValence: {tp['valence_delta']:+.2f})
                    </div>
                    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: Batch Text Stream & Dataset
# -------------------------------------------------------------
with tab3:
    st.markdown("#### Paste multiple messages (one per line) or upload a CSV:")
    
    batch_raw = st.text_area(
        "Batch Text Lines",
        value="""I love this product so much, it changed my workflow! ❤️
The packaging arrived completely crushed and unusable. Worst experience ever.
The delivery was on time, standard service.
I am so anxious about my exam results tomorrow morning...
Unbelievable quality, absolutely stunning design! 🎉
This app keeps crashing every time I try to save. Very annoying.
Thank you for your quick and kind assistance! 🙏""",
        height=140,
        key="text_batch_input"
    )

    if batch_raw.strip():
        lines = [line.strip() for line in batch_raw.split("\n") if line.strip()]
        results, df = conv_analyzer.analyze_batch_messages(lines)

        st.markdown(f"**Processed {len(results)} messages:**")
        
        # Batch Visualization Row
        b_col1, b_col2 = st.columns([5, 7])
        with b_col1:
            st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8; font-weight:600;'>Batch Emotion Distribution</p>", unsafe_allow_html=True)
            emo_counts = df["Dominant Emotion"].value_counts().to_dict()
            st.plotly_chart(render_emotion_distribution_pie(emo_counts), use_container_width=True)

        with b_col2:
            st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8; font-weight:600;'>Batch Dataset Breakdown</p>", unsafe_allow_html=True)
            st.dataframe(df[["Index", "Dominant Emotion", "Confidence (%)", "Valence", "Polarity", "Message"]], use_container_width=True, height=250)

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
st.markdown("### 🛠️ Other Multimodal Studio Modules")
mod1, mod2, mod3 = st.columns(3)
with mod1:
    st.markdown("""
    <div class="es-card" style="height: 100%;">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🎥</div>
        <h3 style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">Live Webcam / Vision Studio</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
            Track 468 3D facial landmarks, micro-expressions, head pose, and action units in real time.
        </p>
    </div>
    """, unsafe_allow_html=True)

with mod2:
    st.markdown("""
    <div class="es-card" style="height: 100%;">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">📁</div>
        <h3 style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">Video & Audio File Analysis</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
            Analyze pre-recorded MP4, WAV, and MP3 files with acoustic prosody and timeline telemetry.
        </p>
    </div>
    """, unsafe_allow_html=True)

with mod3:
    st.markdown("""
    <div class="es-card" style="height: 100%;">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">📑</div>
        <h3 style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">Session History & Reports</h3>
        <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
            Inspect past recording sessions, emotional turning points, and export full telemetry datasets.
        </p>
    </div>
    """, unsafe_allow_html=True)
