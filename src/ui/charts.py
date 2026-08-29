"""Plotly-based interactive data visualization & affective telemetry charts."""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import numpy as np

from config import THEME_COLORS, EMOTION_COLORS
from src.core.types import MultimodalEmotionState, AffectVector, FacialActionUnits


def render_emotion_radar_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Renders a dynamic 8-emotion radar / polar area dial."""
    categories = list(probabilities.keys())
    values = [probabilities.get(c, 0.0) for c in categories]
    # Close the radar polygon loop
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=[c.capitalize() for c in categories],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.35)',
        line=dict(color='#818cf8', width=2),
        marker=dict(size=4, color='#06b6d4'),
        name='Affect Intensity'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                showticklabels=False,
                gridcolor='rgba(255, 255, 255, 0.08)',
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.08)',
                linecolor='rgba(255, 255, 255, 0.15)',
                tickfont=dict(color='#94a3b8', size=11, family='Outfit')
            ),
            bgcolor='rgba(15, 23, 42, 0.5)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=25, r=25, t=25, b=25),
        showlegend=False,
        height=280,
    )
    return fig


def render_affect_quadrant_chart(affect: AffectVector) -> go.Figure:
    """Renders Russell's 2D Circumplex Valence-Arousal Quadrant with live target marker."""
    fig = go.Figure()

    # Background Quadrant Reference Grid
    fig.add_shape(type="rect", x0=-1, y0=0, x1=0, y1=1, fillcolor="rgba(239, 68, 68, 0.07)", line_width=0)   # Angry/Distressed
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, fillcolor="rgba(16, 185, 129, 0.09)", line_width=0)   # Happy/Excited
    fig.add_shape(type="rect", x0=-1, y0=-1, x1=0, y1=0, fillcolor="rgba(59, 130, 246, 0.07)", line_width=0) # Sad/Depressed
    fig.add_shape(type="rect", x0=0, y0=-1, x1=1, y1=0, fillcolor="rgba(6, 182, 212, 0.07)", line_width=0)   # Calm/Content

    # Reference Axis Crosshair
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)

    # Current Live Affect Point
    fig.add_trace(go.Scatter(
        x=[affect.valence],
        y=[affect.arousal],
        mode='markers+text',
        marker=dict(
            size=18,
            color='#6366f1',
            line=dict(width=3, color='#38bdf8'),
            symbol='circle'
        ),
        text=[f" V: {affect.valence:+.2f}<br> A: {affect.arousal:+.2f}"],
        textposition="top right",
        textfont=dict(color='#f8fafc', size=11, family='JetBrains Mono'),
        name="Current Affect"
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="◄ Negative (Valence) Positive ►", font=dict(color="#94a3b8", size=10)),
            range=[-1.05, 1.05],
            zeroline=False,
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#64748b", size=9),
        ),
        yaxis=dict(
            title=dict(text="◄ Calm (Arousal) Excited ►", font=dict(color="#94a3b8", size=10)),
            range=[-1.05, 1.05],
            zeroline=False,
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#64748b", size=9),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.45)',
        margin=dict(l=30, r=30, t=20, b=30),
        showlegend=False,
        height=280,
    )
    return fig


def render_emotion_timeline_chart(history: List[MultimodalEmotionState]) -> go.Figure:
    """Renders a multi-line time-series graph tracking Valence, Arousal, Engagement, and Attention."""
    fig = go.Figure()

    if not history:
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.45)',
            height=250,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(text="No Stream Data Yet", showarrow=False, font=dict(color="#64748b", size=14))]
        )
        return fig

    timestamps = [i for i in range(len(history))]
    valences = [h.affect.valence for h in history]
    arousals = [h.affect.arousal for h in history]
    engagements = [h.engagement_index for h in history]
    attentions = [h.attention_score for h in history]

    fig.add_trace(go.Scatter(x=timestamps, y=valences, mode='lines', name='Valence', line=dict(color='#10b981', width=2)))
    fig.add_trace(go.Scatter(x=timestamps, y=arousals, mode='lines', name='Arousal', line=dict(color='#f59e0b', width=2)))
    fig.add_trace(go.Scatter(x=timestamps, y=engagements, mode='lines', name='Engagement', line=dict(color='#6366f1', width=2)))
    fig.add_trace(go.Scatter(x=timestamps, y=attentions, mode='lines', name='Attention', line=dict(color='#06b6d4', width=1.5, dash='dot')))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Timeline Samples (Frame Index)", font=dict(color="#94a3b8", size=10)),
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#64748b", size=9)
        ),
        yaxis=dict(
            range=[-1.05, 1.05],
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#64748b", size=9)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.45)',
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94a3b8", size=10)
        ),
        height=260,
    )
    return fig


def render_action_units_bar_chart(aus: FacialActionUnits) -> go.Figure:
    """Renders horizontal bar intensities for key facial action units."""
    au_dict = {
        "AU1 Inner Brow": aus.inner_brow_raiser,
        "AU4 Brow Lowerer": aus.brow_lowerer,
        "AU6 Cheek Raiser": aus.cheek_raiser,
        "AU12 Smile Lip": aus.lip_corner_puller,
        "AU15 Lip Depressor": aus.lip_corner_depressor,
        "AU26 Jaw Drop": aus.jaw_drop,
    }

    labels = list(au_dict.keys())
    values = list(au_dict.values())

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(
            color=values,
            colorscale=[[0, '#312e81'], [0.5, '#6366f1'], [1, '#06b6d4']],
        )
    ))

    fig.update_layout(
        xaxis=dict(range=[0, 1.0], gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#64748b", size=9)),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#94a3b8", size=10, family='Outfit')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.45)',
        margin=dict(l=10, r=20, t=10, b=10),
        height=240,
    )
    return fig
