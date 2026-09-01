"""Reusable UI components, status cards, and layout blocks."""

import streamlit as st
from typing import Any, List, Dict, Optional
from config import APP_NAME, APP_TAGLINE, APP_ICON, THEME_COLORS, EMOTION_COLORS
from src.core.types import MultimodalEmotionState, AffectVector



def render_header(title: str = APP_NAME, subtitle: str = APP_TAGLINE):
    """Renders the standard top hero header."""
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 2.2rem;">{APP_ICON}</span>
            <span class="es-hero-title">{title}</span>
        </div>
        <div class="es-hero-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = None, color: str = "#6366f1"):
    """Renders a sleek glowing glass metric card."""
    delta_html = f'<span style="font-size: 0.75rem; color: #10b981; margin-left: 8px;">{delta}</span>' if delta else ''
    st.markdown(f"""
    <div class="es-metric-box" style="border-top: 2px solid {color};">
        <span class="es-metric-label">{label}</span>
        <div style="display: flex; align-items: baseline;">
            <span class="es-metric-value" style="color: {color};">{value}</span>
            {delta_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_affect_summary_badge(state: MultimodalEmotionState):
    """Renders a prominent pill badge displaying the dominant emotion & confidence."""
    emo = state.dominant_emotion
    color = EMOTION_COLORS.get(emo, "#6366f1")
    conf_pct = int(state.confidence * 100)

    st.markdown(f"""
    <div style="
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid {color}55;
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px {color}22;
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: {color}; box-shadow: 0 0 10px {color};"></div>
            <div>
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Dominant Affect</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc;">{emo.capitalize()}</div>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Confidence</div>
            <div style="font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: {color};">{conf_pct}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_state_indicators(state: MultimodalEmotionState):
    """Renders 3 mini health gauges for Engagement, Attention, and Fatigue."""
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Engagement", f"{int(state.engagement_index * 100)}%", color="#6366f1")
    with c2:
        render_metric_card("Attention", f"{int(state.attention_score * 100)}%", color="#06b6d4")
    with c3:
        fatigue_color = "#ef4444" if state.fatigue_level > 0.6 else "#10b981"
        render_metric_card("Fatigue", f"{int(state.fatigue_level * 100)}%", color=fatigue_color)


def render_chat_bubble(turn: Any, is_right: bool = False):
    """Renders a glassmorphic chat message bubble with affective badge and aura."""
    emo_res = turn.emotion_result
    emo = emo_res.dominant_emotion
    color = EMOTION_COLORS.get(emo, "#6366f1")
    conf_pct = int(emo_res.confidence * 100)
    
    align_style = "margin-left: auto; border-left: 4px solid #6366f1;" if is_right else f"margin-right: auto; border-left: 4px solid {color};"
    bg_gradient = "linear-gradient(135deg, rgba(30, 27, 75, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%)" if is_right else "linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(30, 41, 59, 0.4) 100%)"

    time_html = f'<span style="font-size: 0.75rem; color: #64748b; margin-left: 8px;">{turn.timestamp_str}</span>' if turn.timestamp_str else ''
    speaker_icon = "👤" if is_right else "🗣️"

    st.markdown(f"""
    <div style="
        max-width: 88%;
        {align_style}
        background: {bg_gradient};
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        border-right: 1px solid rgba(255,255,255,0.06);
        border-bottom: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
            <div style="font-weight: 600; font-size: 0.85rem; color: #e2e8f0; display: flex; align-items: center; gap: 6px;">
                <span>{speaker_icon}</span>
                <span>{turn.speaker}</span>
                {time_html}
            </div>
            <div style="
                background: {color}22;
                border: 1px solid {color}66;
                color: {color};
                padding: 2px 8px;
                border-radius: 9999px;
                font-size: 0.7rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            ">
                ● {emo} ({conf_pct}%)
            </div>
        </div>
        <div style="color: #f1f5f9; font-size: 0.95rem; line-height: 1.5; margin-bottom: 0.3rem;">
            {turn.text}
        </div>
        <div style="display: flex; gap: 12px; font-size: 0.75rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; padding-top: 4px; border-top: 1px solid rgba(255,255,255,0.05);">
            <span>V: <b style="color: {'#10b981' if emo_res.affect.valence >= 0 else '#ef4444'}">{emo_res.affect.valence:+.2f}</b></span>
            <span>A: <b style="color: #f59e0b">{emo_res.affect.arousal:+.2f}</b></span>
            <span>Pol: <b style="color: #818cf8">{emo_res.polarity:+.2f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_token_explainability_box(tokens: list):
    """Renders token-level explainability showing which words influenced the emotion."""
    if not tokens:
        return

    html_parts = []
    for t in tokens:
        if t.emotion_influence in ("neutral", "negation", "intensifier", "diminisher") and t.score < 0.2:
            html_parts.append(f'<span style="color: #cbd5e1; margin-right: 6px; font-size: 0.95rem;">{t.token}</span>')
        else:
            base_color = EMOTION_COLORS.get(t.emotion_influence.split()[0], "#818cf8")
            badge_title = f"{t.emotion_influence} (score: {t.score:.2f})"
            html_parts.append(f"""
            <span title="{badge_title}" style="
                background: {base_color}25;
                color: #ffffff;
                border: 1px solid {base_color}88;
                padding: 2px 7px;
                border-radius: 6px;
                font-weight: 600;
                margin-right: 6px;
                font-size: 0.95rem;
                display: inline-block;
                margin-bottom: 4px;
            ">
                {t.token} <small style="color: {base_color}; font-size: 0.7rem;">● {t.emotion_influence}</small>
            </span>
            """)

    tokens_html = "".join(html_parts)
    st.markdown(f"""
    <div style="
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-top: 0.6rem;
        line-height: 1.8;
    ">
        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 0.4rem;">
            🔍 Token Salience & Affective Triggers
        </div>
        {tokens_html}
    </div>
    """, unsafe_allow_html=True)


def render_empathy_advice_card(advice: str, dominant_emotion: str):
    """Renders actionable empathy and communication suggestions."""
    color = EMOTION_COLORS.get(dominant_emotion, "#6366f1")
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.35) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid {color}44;
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        margin-top: 0.75rem;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    ">
        <div style="font-size: 1.4rem;">💡</div>
        <div>
            <div style="font-size: 0.75rem; color: {color}; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">
                Empathy & Response Recommendation
            </div>
            <div style="color: #e2e8f0; font-size: 0.9rem; margin-top: 0.2rem; line-height: 1.45;">
                {advice}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

