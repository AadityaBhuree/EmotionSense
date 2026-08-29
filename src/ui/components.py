"""Reusable UI components, status cards, and layout blocks."""

import streamlit as st
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
