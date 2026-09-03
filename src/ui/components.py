"""Reusable high-density UI components, telemetry cards, and layout blocks."""

import streamlit as st
from typing import Any, List, Dict, Optional
from config import APP_NAME, APP_TAGLINE, APP_ICON, THEME_COLORS, EMOTION_COLORS
from src.core.types import MultimodalEmotionState, AffectVector


def render_header(title: str = APP_NAME, subtitle: str = APP_TAGLINE):
    """Renders a precision workstation console header bar with active status telemetry."""
    st.markdown(f"""
    <div class="es-console-bar">
        <div class="es-console-brand">
            <span style="font-size: 1.8rem; line-height: 1;">{APP_ICON}</span>
            <div>
                <div class="es-console-title">
                    {title}
                    <span class="es-pill es-pill-active" style="font-size: 0.68rem; padding: 2px 7px;">
                        <span class="es-live-beacon"></span> ONLINE
                    </span>
                </div>
                <div class="es-console-sub">{subtitle}</div>
            </div>
        </div>
        <div class="es-telemetry-cluster">
            <span style="color: var(--text-muted);">LATENCY:</span>
            <span style="color: #10b981; font-weight: 600;">&lt; 4.8ms</span>
            <span style="color: var(--border-color);">|</span>
            <span style="color: var(--text-muted);">FRAMEWORK:</span>
            <span style="color: #94a3b8; font-weight: 500;">Affective-VAD v2.0</span>
            <span style="color: var(--border-color);">|</span>
            <span class="es-pill es-pill-idle" style="font-size: 0.68rem;">TELEMETRY ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    color: Optional[str] = None,
    subtext: Optional[str] = None
):
    """Renders a high-density, calibrated instrument metric gauge."""
    accent_color = color or THEME_COLORS.get("primary", "#3b82f6")
    delta_html = (
        f'<span style="font-size: 0.72rem; color: #10b981; font-family: \'JetBrains Mono\', monospace; margin-left: 6px;">{delta}</span>'
        if delta else ''
    )
    subtext_html = (
        f'<div class="es-metric-meta">{subtext}</div>'
        if subtext else ''
    )

    st.markdown(f"""
    <div class="es-metric-box" style="border-left: 3px solid {accent_color};">
        <div class="es-metric-label">
            <span>{label}</span>
            <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: {accent_color};"></span>
        </div>
        <div style="display: flex; align-items: baseline; justify-content: space-between; margin-top: 0.15rem;">
            <span class="es-metric-value" style="color: {accent_color};">{value}</span>
            {delta_html}
        </div>
        {subtext_html}
    </div>
    """, unsafe_allow_html=True)


def render_affect_summary_badge(state: MultimodalEmotionState):
    """Renders a precision instrument gauge displaying dominant emotion, confidence, and VAD coordinates."""
    emo = state.dominant_emotion
    color = EMOTION_COLORS.get(emo, THEME_COLORS.get("primary", "#3b82f6"))
    conf_pct = int(state.confidence * 100)

    val = state.affect.valence
    aro = state.affect.arousal
    dom = state.affect.dominance

    st.markdown(f"""
    <div style="
        background: var(--surface-card);
        border: 1px solid var(--border-color);
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background: {color};"></div>
            <div>
                <div style="font-size: 0.7rem; color: var(--text-sub); text-transform: uppercase; letter-spacing: 0.06em; font-family: 'JetBrains Mono', monospace;">
                    Dominant Affect
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">
                    {emo.capitalize()}
                </div>
            </div>
        </div>
        
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: var(--text-sub);">
                <div>V: <b style="color: {'#10b981' if val >= 0 else '#ef4444'}">{val:+.2f}</b> | A: <b style="color: #f59e0b">{aro:+.2f}</b></div>
                <div style="color: var(--text-muted); font-size: 0.7rem;">D: {dom:+.2f}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.68rem; color: var(--text-sub); text-transform: uppercase; letter-spacing: 0.06em;">Confidence</div>
                <div style="font-size: 1.25rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: {color};">
                    {conf_pct}%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_state_indicators(state: MultimodalEmotionState):
    """Renders 3 calibrated health gauges for Engagement, Attention, and Cognitive Fatigue."""
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "Engagement Index",
            f"{int(state.engagement_index * 100)}%",
            delta="Nominal" if state.engagement_index > 0.5 else "Low",
            color="#3b82f6"
        )
    with c2:
        render_metric_card(
            "Attention Score",
            f"{int(state.attention_score * 100)}%",
            delta="Focused" if state.attention_score > 0.6 else "Distracted",
            color="#0ea5e9"
        )
    with c3:
        fatigue_color = "#ef4444" if state.fatigue_level > 0.6 else ("#f59e0b" if state.fatigue_level > 0.4 else "#10b981")
        fatigue_status = "Elevated" if state.fatigue_level > 0.6 else ("Moderate" if state.fatigue_level > 0.4 else "Low")
        render_metric_card(
            "Cognitive Fatigue",
            f"{int(state.fatigue_level * 100)}%",
            delta=fatigue_status,
            color=fatigue_color
        )


def render_chat_bubble(turn: Any, is_right: bool = False):
    """Renders a precision diagnostic chat message turn with micro VAD coordinates and dominant affect pill."""
    emo_res = turn.emotion_result
    emo = emo_res.dominant_emotion
    color = EMOTION_COLORS.get(emo, THEME_COLORS.get("primary", "#3b82f6"))
    conf_pct = int(emo_res.confidence * 100)
    
    border_indicator = f"border-left: 3px solid {color};"
    bg_surface = "#141a27" if is_right else "#101520"
    speaker_badge = "USER" if is_right else (turn.speaker if hasattr(turn, "speaker") else "AGENT")
    speaker_color = "#38bdf8" if is_right else "#a855f7"

    time_html = f'<span style="font-size: 0.7rem; color: #526079; margin-left: 8px; font-family: \'JetBrains Mono\', monospace;">{turn.timestamp_str}</span>' if getattr(turn, "timestamp_str", None) else ''

    st.markdown(f"""
    <div style="
        max-width: 92%;
        margin-left: {'auto' if is_right else '0'};
        margin-right: {'0' if is_right else 'auto'};
        background: {bg_surface};
        border: 1px solid var(--border-color);
        {border_indicator}
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
            <div style="font-weight: 700; font-size: 0.76rem; color: {speaker_color}; font-family: 'JetBrains Mono', monospace; display: flex; align-items: center; gap: 4px;">
                <span>[{speaker_badge}]</span>
                {time_html}
            </div>
            <div style="
                background: {color}18;
                border: 1px solid {color}55;
                color: {color};
                padding: 1px 7px;
                border-radius: 3px;
                font-size: 0.68rem;
                font-family: 'JetBrains Mono', monospace;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            ">
                ● {emo} {conf_pct}%
            </div>
        </div>
        
        <div style="color: #f1f5f9; font-size: 0.92rem; line-height: 1.5; margin-bottom: 0.4rem;">
            {turn.text}
        </div>
        
        <div style="
            display: flex;
            gap: 14px;
            font-size: 0.72rem;
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
            padding-top: 0.35rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        ">
            <span>Valence: <b style="color: {'#10b981' if emo_res.affect.valence >= 0 else '#ef4444'}">{emo_res.affect.valence:+.2f}</b></span>
            <span>Arousal: <b style="color: #f59e0b">{emo_res.affect.arousal:+.2f}</b></span>
            <span>Polarity: <b style="color: #38bdf8">{emo_res.polarity:+.2f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_token_explainability_box(tokens: list):
    """Renders a precision token salience spectrum showing linguistic affective weights."""
    if not tokens:
        return

    html_parts = []
    for t in tokens:
        if t.emotion_influence in ("neutral", "negation", "intensifier", "diminisher") and t.score < 0.2:
            html_parts.append(
                f'<span style="color: #94a3b8; margin-right: 6px; font-size: 0.88rem; font-family: \'JetBrains Mono\', monospace;">{t.token}</span>'
            )
        else:
            emo_name = t.emotion_influence.split()[0]
            base_color = EMOTION_COLORS.get(emo_name, "#3b82f6")
            badge_title = f"{t.emotion_influence} | weight: {t.score:.2f}"
            html_parts.append(f"""
            <span title="{badge_title}" style="
                background: {base_color}18;
                color: #f8fafc;
                border: 1px solid {base_color}66;
                padding: 2px 7px;
                border-radius: 4px;
                font-weight: 600;
                margin-right: 6px;
                font-size: 0.85rem;
                font-family: 'JetBrains Mono', monospace;
                display: inline-block;
                margin-bottom: 4px;
            ">
                {t.token} <small style="color: {base_color}; font-size: 0.68rem;">[{t.emotion_influence[:4]}:{t.score:.1f}]</small>
            </span>
            """)

    tokens_html = "".join(html_parts)
    st.markdown(f"""
    <div style="
        background: var(--surface-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-top: 0.65rem;
        line-height: 1.8;
    ">
        <div style="font-size: 0.72rem; color: var(--text-sub); text-transform: uppercase; font-family: 'JetBrains Mono', monospace; font-weight: 600; letter-spacing: 0.06em; margin-bottom: 0.4rem;">
            ⌖ Token Salience Spectrum & Affective Weights
        </div>
        {tokens_html}
    </div>
    """, unsafe_allow_html=True)


def render_empathy_advice_card(advice: str, dominant_emotion: str):
    """Renders structured clinical/behavioral recommendation with calibrated action points."""
    color = EMOTION_COLORS.get(dominant_emotion, THEME_COLORS.get("primary", "#3b82f6"))
    st.markdown(f"""
    <div style="
        background: var(--surface-raised);
        border: 1px solid var(--border-color);
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-top: 0.75rem;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    ">
        <div style="font-size: 1.3rem; line-height: 1.2;">⚡</div>
        <div>
            <div style="font-size: 0.72rem; color: {color}; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; font-weight: 700; letter-spacing: 0.06em;">
                Clinical Empathy & Response Recommendation
            </div>
            <div style="color: #e2e8f0; font-size: 0.88rem; margin-top: 0.25rem; line-height: 1.5;">
                {advice}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
