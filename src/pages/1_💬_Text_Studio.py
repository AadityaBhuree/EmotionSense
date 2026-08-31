"""Page 1: Dedicated Text Message & Conversational Affective Intelligence Studio."""

import streamlit as st
import time
import pandas as pd
import json

from config import THEME_COLORS, EMOTION_COLORS
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
from src.utils.session_manager import SessionManager

# Page Configuration
st.set_page_config(page_title="Text Emotion Studio | EmotionSense", page_icon="💬", layout="wide")
inject_glassmorphic_styles()

render_header("Text Emotion & Dialogue Studio", "Deep Affective NLP, Conversational Trajectory & Token Salience")

# Initialize Session State
if "text_classifier" not in st.session_state:
    st.session_state.text_classifier = TextEmotionClassifier()
if "conversation_analyzer" not in st.session_state:
    st.session_state.conversation_analyzer = ConversationAffectAnalyzer(st.session_state.text_classifier)
if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()

clf = st.session_state.text_classifier
conv = st.session_state.conversation_analyzer

# Studio Mode Radio
mode = st.radio(
    "Select Studio Analysis Mode",
    ["📝 Single Message & Live Salience", "🗨️ Multi-turn Dialogue Transcript", "📊 Batch File / Multi-Line Stream"],
    horizontal=True
)

st.markdown("---")

if mode == "📝 Single Message & Live Salience":
    st.markdown("### 📝 Single Text Message Affect Decoder")
    
    col_in, col_opts = st.columns([8, 4])
    with col_in:
        text_input = st.text_area(
            "Input Message Text",
            value="I am absolutely thrilled and excited about our new project launch!! The results are fantastic! 🚀🎉",
            height=120,
            placeholder="Type or paste any message..."
        )
    with col_opts:
        st.markdown("<div class='es-card'>", unsafe_allow_html=True)
        st.markdown("<b style='color: #f8fafc; font-size: 0.9rem;'>⚡ Quick Presets</b>", unsafe_allow_html=True)
        if st.button("🎉 Joyful Milestone", use_container_width=True):
            text_input = "We finally hit 1 million users today! So proud of the entire engineering team! 🥳🍾"
        if st.button("😡 Outraged Customer", use_container_width=True):
            text_input = "This is the worst customer service ever. You stole my money and refuse to answer! Absolute scam!"
        if st.button("😰 Anxious / Stressed", use_container_width=True):
            text_input = "I'm terrified of failing the final exam tomorrow, having panic attacks and cannot sleep."
        if st.button("💙 Heartfelt Empathy", use_container_width=True):
            text_input = "I am so deeply sorry for your loss. Please know that we are all here to support you."
        st.markdown("</div>", unsafe_allow_html=True)

    if text_input.strip():
        res = clf.analyze_text(text_input)
        dom = res.dominant_emotion
        color = EMOTION_COLORS.get(dom, "#6366f1")

        # Metric Tiles
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Dominant Affect", dom.upper(), delta=f"{int(res.confidence * 100)}% Conf", color=color)
        with m2:
            render_metric_card("Valence (Mood)", f"{res.affect.valence:+.2f}", delta="Negative ◄► Positive", color="#10b981" if res.affect.valence >= 0 else "#ef4444")
        with m3:
            render_metric_card("Arousal (Energy)", f"{res.affect.arousal:+.2f}", delta="Calm ◄► Excited", color="#f59e0b")
        with m4:
            render_metric_card("Dominance", f"{res.affect.dominance:+.2f}", delta="Submissive ◄► Empowered", color="#06b6d4")

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualizations
        v1, v2, v3 = st.columns([4, 4, 4])
        with v1:
            st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8; font-weight:600;'>Probability Distribution</p>", unsafe_allow_html=True)
            st.plotly_chart(render_emotion_horizontal_bars(res.probabilities), use_container_width=True)
        with v2:
            st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8; font-weight:600;'>8-Emotion Polar Radar</p>", unsafe_allow_html=True)
            st.plotly_chart(render_emotion_radar_chart(res.probabilities), use_container_width=True)
        with v3:
            st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8; font-weight:600;'>Russell's 2D VAD Circumplex</p>", unsafe_allow_html=True)
            st.plotly_chart(render_affect_quadrant_chart(res.affect), use_container_width=True)

        # Salience & Empathy
        e1, e2 = st.columns([6, 6])
        with e1:
            render_token_explainability_box(res.salience_tokens)
        with e2:
            render_empathy_advice_card(res.empathy_advice, dom)
            
            # AI Empathy Auto-Reply Assistant
            st.markdown("""
            <div style="margin-top: 0.75rem; padding: 0.85rem 1rem; background: rgba(15, 23, 42, 0.55); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.75rem; color: #818cf8; text-transform: uppercase; font-weight: 700; margin-bottom: 6px;">
                    ✨ Recommended Empathetic Reply Templates
                </div>
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
                st.markdown(f'<div style="background: rgba(30, 41, 59, 0.45); padding: 6px 10px; border-radius: 6px; margin-bottom: 5px; font-size: 0.85rem; color: #e2e8f0; border-left: 2px solid #6366f1;">💬 {r}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


elif mode == "🗨️ Multi-turn Dialogue Transcript":
    st.markdown("### 🗨️ Conversational Flow & Escalation Studio")

    default_transcript = """[10:15 AM] User: Hello, I have an urgent issue with my cloud deployment.
[10:16 AM] Agent: Hi! I am here to help you. What error code are you encountering?
[10:17 AM] User: It says 500 Internal Server Error and all our customer checkout lines are dead!! I am losing revenue every minute!!
[10:18 AM] Agent: I understand how critical this is. I am escalating directly to our Tier 3 infrastructure engineers right now.
[10:19 AM] User: Please hurry, this is terrifying for our business.
[10:21 AM] Agent: The server route was unblocked and traffic is fully normal now. We have applied a credit to your account.
[10:22 AM] User: Wow, thank goodness! It's working perfectly now. Thank you so much for the rapid resolution! 🎉"""

    transcript_in = st.text_area("Paste Chat Transcript", value=default_transcript, height=180)

    if transcript_in.strip():
        summary = conv.parse_and_analyze_transcript(transcript_in)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_metric_card("Total Turns", str(summary.total_turns), delta=f"{len(summary.speakers)} Speakers", color="#6366f1")
        with k2:
            esc_col = "#10b981" if summary.escalation_risk == "Low" else ("#f59e0b" if summary.escalation_risk == "Moderate" else "#ef4444")
            render_metric_card("Escalation Risk", summary.escalation_risk, delta="Conflict Watchdog", color=esc_col)
        with k3:
            render_metric_card("Empathy / Rapport", f"{int(summary.rapport_empathy_score * 100)}%", delta="Mirroring Synchrony", color="#06b6d4")
        with k4:
            render_metric_card("Turning Points", str(len(summary.turning_points)), delta="Inflection Shifts", color="#f59e0b")

        st.markdown("<br>", unsafe_allow_html=True)

        col_chat, col_flow = st.columns([5, 6])
        with col_chat:
            st.markdown("#### 💬 Message Stream with Affect Aura")
            for turn in summary.turns:
                is_right = (turn.speaker == summary.speakers[-1]) if len(summary.speakers) > 1 else False
                render_chat_bubble(turn, is_right=is_right)

        with col_flow:
            st.markdown("#### 📈 Conversational Mood & Energy Trajectory")
            st.plotly_chart(render_conversation_flow_chart(summary.emotional_trajectory), use_container_width=True)

            if summary.turning_points:
                st.markdown("##### ⚡ Turning Points Detected")
                for tp in summary.turning_points:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.4); border-left: 3px solid #f59e0b; padding: 6px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 0.85rem;">
                        <b>Turn #{tp['turn_index']} ({tp['speaker']})</b>: <code>{tp['from_emotion']}</code> ➔ <code style="color: #f59e0b;">{tp['to_emotion']}</code> (ΔValence: {tp['valence_delta']:+.2f})
                    </div>
                    """, unsafe_allow_html=True)


elif mode == "📊 Batch File / Multi-Line Stream":
    st.markdown("### 📊 Batch Text Affect Intelligence & File Upload")

    up_file = st.file_uploader("Upload CSV or TXT File", type=["csv", "txt"])
    
    batch_text = ""
    if up_file is not None:
        if up_file.name.endswith(".csv"):
            raw_df = pd.read_csv(up_file)
            # Find first string column
            str_cols = [c for c in raw_df.columns if raw_df[c].dtype == "object"]
            if str_cols:
                batch_text = "\n".join(raw_df[str_cols[0]].dropna().astype(str).tolist())
            else:
                batch_text = "\n".join(raw_df.iloc[:, 0].dropna().astype(str).tolist())
        else:
            batch_text = up_file.read().decode("utf-8")
    else:
        batch_text = st.text_area(
            "Or Paste Lines of Text Below (One Message Per Line)",
            value="""I am so thrilled with this achievement! 🎉
The package was damaged and customer service was unresponsive. 😡
Regular update, nothing remarkable.
Anxious about the upcoming review tomorrow...
Thank you for being so supportive and kind. ❤️
Utterly disgusting experience at the restaurant. 🤮
Astonishing performance, way beyond expectations! 🚀""",
            height=150
        )

    if batch_text.strip():
        lines = [line.strip() for line in batch_text.split("\n") if line.strip()]
        results, df = conv.analyze_batch_messages(lines)

        st.markdown(f"**Analyzed {len(results)} distinct items:**")
        
        bc1, bc2 = st.columns([5, 7])
        with bc1:
            st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8; font-weight:600;'>Batch Affect Distribution</p>", unsafe_allow_html=True)
            counts = df["Dominant Emotion"].value_counts().to_dict()
            st.plotly_chart(render_emotion_distribution_pie(counts), use_container_width=True)

        with bc2:
            st.markdown("<p style='text-align:center; font-size:0.85rem; color:#94a3b8; font-weight:600;'>Telemetry Data Table</p>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=270)

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
