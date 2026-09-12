"""Plotly-based scientific interactive visualization & neuro-affective telemetry charts."""

import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

from config import EMOTION_COLORS
from src.core.types import MultimodalEmotionState, AffectVector, FacialActionUnits


# Instrument Chart Styling Constants
CHART_BG = "rgba(18, 22, 34, 0.6)"
GRID_COLOR = "rgba(255, 255, 255, 0.05)"
AXIS_COLOR = "#94a3b8"
FONT_DISPLAY = "Plus Jakarta Sans, -apple-system, sans-serif"
FONT_MONO = "JetBrains Mono, monospace"


def render_emotion_radar_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Renders a precision wireframe 8-emotion polar radar dial."""
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
        fillcolor='rgba(59, 130, 246, 0.22)',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=5, color='#60a5fa'),
        name='Affect Intensity',
        hoverinfo='theta+r'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                showticklabels=False,
                gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR,
            ),
            angularaxis=dict(
                gridcolor=GRID_COLOR,
                linecolor='rgba(255, 255, 255, 0.12)',
                tickfont=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)
            ),
            bgcolor=CHART_BG
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=28, r=28, t=25, b=25),
        showlegend=False,
        height=270,
    )
    return fig


def render_affect_quadrant_chart(
    affect: AffectVector,
    history_affects: Optional[List[AffectVector]] = None
) -> go.Figure:
    """Renders Russell's 2D Circumplex (Valence x Arousal) with quadrant zones and trajectory trail."""
    fig = go.Figure()

    # Subtle Quadrant Background Zones
    fig.add_shape(type="rect", x0=-1, y0=0, x1=0, y1=1, fillcolor="rgba(239, 68, 68, 0.05)", line_width=0)   # Agitated / Hostile
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, fillcolor="rgba(16, 185, 129, 0.06)", line_width=0)   # Ebullient / Excited
    fig.add_shape(type="rect", x0=-1, y0=-1, x1=0, y1=0, fillcolor="rgba(59, 130, 246, 0.05)", line_width=0) # Depressed / Withdrawn
    fig.add_shape(type="rect", x0=0, y0=-1, x1=1, y1=0, fillcolor="rgba(14, 165, 233, 0.05)", line_width=0)   # Serene / Composed

    # Reference Neutral Baseline Zone (Circle at center)
    fig.add_shape(
        type="circle",
        x0=-0.22, y0=-0.22, x1=0.22, y1=0.22,
        line=dict(color="rgba(255, 255, 255, 0.1)", dash="dot", width=1),
        fillcolor="rgba(255, 255, 255, 0.015)"
    )

    # Reference Axis Crosshair
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.12)", line_width=1)
    fig.add_vline(x=0, line_dash="solid", line_color="rgba(255,255,255,0.12)", line_width=1)

    # Quadrant Label Annotations
    quadrant_annotations = [
        dict(x=0.85, y=0.92, text="EBULLIENT", showarrow=False, font=dict(color="#10b981", size=8, family=FONT_MONO)),
        dict(x=-0.82, y=0.92, text="AGITATED", showarrow=False, font=dict(color="#ef4444", size=8, family=FONT_MONO)),
        dict(x=-0.80, y=-0.92, text="WITHDRAWN", showarrow=False, font=dict(color="#3b82f6", size=8, family=FONT_MONO)),
        dict(x=0.85, y=-0.92, text="SERENE", showarrow=False, font=dict(color="#0ea5e9", size=8, family=FONT_MONO)),
    ]
    for ann in quadrant_annotations:
        fig.add_annotation(**ann)

    # Historical Trajectory Trail (fading dots)
    if history_affects and len(history_affects) > 1:
        recent_pts = history_affects[-8:]
        trail_x = [pt.valence for pt in recent_pts[:-1]]
        trail_y = [pt.arousal for pt in recent_pts[:-1]]
        if trail_x:
            fig.add_trace(go.Scatter(
                x=trail_x,
                y=trail_y,
                mode='lines+markers',
                line=dict(color='rgba(96, 165, 250, 0.35)', width=1, dash='dot'),
                marker=dict(size=5, color='rgba(96, 165, 250, 0.45)'),
                hoverinfo='skip',
                name='Vector Trail'
            ))

    # Current Live Affect Coordinate (Tactile Reticule)
    fig.add_trace(go.Scatter(
        x=[affect.valence],
        y=[affect.arousal],
        mode='markers+text',
        marker=dict(
            size=14,
            color='#3b82f6',
            line=dict(width=2, color='#f1f5f9'),
            symbol='cross'
        ),
        text=[f"[{affect.valence:+.2f}, {affect.arousal:+.2f}]"],
        textposition="top right",
        textfont=dict(color='#f1f5f9', size=10, family=FONT_MONO),
        name="Current Coordinate",
        hoverinfo='text'
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="◄ NEGATIVE (Valence) POSITIVE ►", font=dict(color=AXIS_COLOR, size=9, family=FONT_MONO)),
            range=[-1.05, 1.05],
            zeroline=False,
            gridcolor=GRID_COLOR,
            tickfont=dict(color="#526079", size=8, family=FONT_MONO),
        ),
        yaxis=dict(
            title=dict(text="◄ CALM (Arousal) EXCITED ►", font=dict(color=AXIS_COLOR, size=9, family=FONT_MONO)),
            range=[-1.05, 1.05],
            zeroline=False,
            gridcolor=GRID_COLOR,
            tickfont=dict(color="#526079", size=8, family=FONT_MONO),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=30, r=30, t=15, b=25),
        showlegend=False,
        height=270,
    )
    return fig


def render_emotion_timeline_chart(history: List[MultimodalEmotionState]) -> go.Figure:
    """Renders a multi-line time-series telemetry graph tracking Valence, Arousal, Engagement, and Attention."""
    fig = go.Figure()

    if not history:
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=CHART_BG,
            height=250,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(text="AWAITING STREAM TELEMETRY...", showarrow=False, font=dict(color="#526079", size=12, family=FONT_MONO))]
        )
        return fig

    timestamps = list(range(len(history)))
    valences = [h.affect.valence for h in history]
    arousals = [h.affect.arousal for h in history]
    engagements = [h.engagement_index for h in history]
    attentions = [h.attention_score for h in history]

    fig.add_trace(go.Scatter(x=timestamps, y=valences, mode='lines', name='Valence', line=dict(color='#10b981', width=1.75)))
    fig.add_trace(go.Scatter(x=timestamps, y=arousals, mode='lines', name='Arousal', line=dict(color='#f59e0b', width=1.75)))
    fig.add_trace(go.Scatter(x=timestamps, y=engagements, mode='lines', name='Engagement', line=dict(color='#3b82f6', width=1.75)))
    fig.add_trace(go.Scatter(x=timestamps, y=attentions, mode='lines', name='Attention', line=dict(color='#0ea5e9', width=1.5, dash='dot')))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Stream Sample Index", font=dict(color=AXIS_COLOR, size=9, family=FONT_MONO)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color="#526079", size=8, family=FONT_MONO)
        ),
        yaxis=dict(
            range=[-1.05, 1.05],
            gridcolor=GRID_COLOR,
            tickfont=dict(color="#526079", size=8, family=FONT_MONO)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=AXIS_COLOR, size=9, family=FONT_MONO)
        ),
        height=250,
    )
    return fig


def render_action_units_bar_chart(aus: FacialActionUnits) -> go.Figure:
    """Renders calibrated horizontal bar intensities for FACS Facial Action Units with activation threshold."""
    au_dict = {
        "AU01 Inner Brow": getattr(aus, "inner_brow_raiser", 0.0),
        "AU04 Brow Lower": getattr(aus, "brow_lowerer", 0.0),
        "AU06 Cheek Raise": getattr(aus, "cheek_raiser", 0.0),
        "AU12 Lip Corner": getattr(aus, "lip_corner_puller", 0.0),
        "AU15 Lip Depress": getattr(aus, "lip_corner_depressor", 0.0),
        "AU20 Lip Stretch": getattr(aus, "lip_stretcher", 0.0) if hasattr(aus, "lip_stretcher") else 0.0,
        "AU25 Lips Part": getattr(aus, "lips_part", 0.0) if hasattr(aus, "lips_part") else 0.0,
        "AU26 Jaw Drop": getattr(aus, "jaw_drop", 0.0),
    }

    labels = list(au_dict.keys())
    values = list(au_dict.values())

    # Activation threshold marker
    bar_colors = ["#ef4444" if v > 0.65 else ("#f59e0b" if v > 0.4 else "#3b82f6") for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color='rgba(255,255,255,0.08)', width=1)),
        text=[f"{v:.2f}" for v in values],
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=9, family=FONT_MONO)
    ))

    # Threshold guideline at 0.5 activation
    fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(245, 158, 11, 0.4)", line_width=1)

    fig.update_layout(
        xaxis=dict(range=[0, 1.15], gridcolor=GRID_COLOR, tickfont=dict(color="#526079", size=8, family=FONT_MONO)),
        yaxis=dict(autorange="reversed", tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=10, r=20, t=10, b=10),
        height=240,
    )
    return fig


def render_conversation_flow_chart(trajectory: List[Dict[str, Any]]) -> go.Figure:
    """Renders turn-by-turn conversational affect trajectory tracking Valence, Arousal, and Conflict Risk."""
    fig = go.Figure()

    if not trajectory:
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=CHART_BG,
            height=280,
            annotations=[dict(text="AWAITING DIALOGUE TURNS...", showarrow=False, font=dict(color="#526079", size=12, family=FONT_MONO))]
        )
        return fig

    turns = [t["turn"] for t in trajectory]
    valences = [t["valence"] for t in trajectory]
    arousals = [t["arousal"] for t in trajectory]
    escalations = [t.get("escalation_score", 0.0) for t in trajectory]
    hover_texts = [
        f"Turn #{t['turn']} | {t['speaker']}<br>{t['text'][:35]}...<br>Emotion: {t['dominant_emotion'].upper()} ({int(t['confidence']*100)}%)"
        for t in trajectory
    ]

    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.12)", line_width=1)
    
    fig.add_trace(go.Scatter(
        x=turns,
        y=valences,
        mode='lines+markers',
        name='Valence (Mood)',
        line=dict(color='#10b981', width=2),
        marker=dict(size=6, color='#34d399'),
        text=hover_texts,
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=turns,
        y=arousals,
        mode='lines+markers',
        name='Arousal (Energy)',
        line=dict(color='#f59e0b', width=1.75, dash='dot'),
        marker=dict(size=5, color='#fbbf24', symbol='diamond'),
        text=hover_texts,
        hoverinfo='text'
    ))

    fig.add_trace(go.Bar(
        x=turns,
        y=escalations,
        name='Conflict Risk',
        marker=dict(color='rgba(239, 68, 68, 0.4)', line=dict(color='#ef4444', width=1)),
        hoverinfo='skip',
        opacity=0.7
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Dialogue Turn Sequence", font=dict(color=AXIS_COLOR, size=9, family=FONT_MONO)),
            tickmode='linear',
            tick0=1,
            dtick=1,
            gridcolor=GRID_COLOR,
            tickfont=dict(color="#526079", size=8, family=FONT_MONO)
        ),
        yaxis=dict(
            range=[-1.1, 1.1],
            title=dict(text="◄ NEGATIVE | POSITIVE ►", font=dict(color=AXIS_COLOR, size=8, family=FONT_MONO)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color="#526079", size=8, family=FONT_MONO)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=20, r=20, t=25, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=AXIS_COLOR, size=9, family=FONT_MONO)
        ),
        height=280,
    )
    return fig


def render_emotion_distribution_pie(distribution: Dict[str, float]) -> go.Figure:
    """Renders a precision donut distribution chart."""
    labels = [k.capitalize() for k in distribution.keys()]
    values = list(distribution.values())
    colors = [EMOTION_COLORS.get(k.lower(), "#3b82f6") for k in distribution.keys()]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color='#0b0e14', width=1.5)),
        textinfo='percent+label',
        textfont=dict(color='#f1f5f9', size=9, family=FONT_MONO),
        hoverinfo='label+percent+value',
    )])

    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=240,
    )
    return fig


def render_emotion_horizontal_bars(probabilities: Dict[str, float], top_n: int = 6) -> go.Figure:
    """Renders ranked horizontal bars for predicted emotion probabilities."""
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    sorted_items.reverse()  # For bottom-to-top rendering

    labels = [k.capitalize() for k, v in sorted_items]
    values = [round(v * 100, 1) for k, v in sorted_items]
    colors = [EMOTION_COLORS.get(k.lower(), "#3b82f6") for k, v in sorted_items]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=9, family=FONT_MONO),
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.08)', width=1)),
    ))

    fig.update_layout(
        xaxis=dict(
            range=[0, max(values, default=100) + 20],
            showticklabels=False,
            gridcolor=GRID_COLOR
        ),
        yaxis=dict(tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_DISPLAY)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=10, r=25, t=10, b=10),
        height=230,
    )
    return fig


def render_dual_track_multimodal_timeline(history: List[MultimodalEmotionState]) -> go.Figure:
    """Renders synchronized dual-track timeline: Visual/Fused Affect vs Acoustic Dynamics."""
    from plotly.subplots import make_subplots

    if not history:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=CHART_BG,
            height=340,
            annotations=[dict(text="AWAITING AUDIOVISUAL STREAM...", showarrow=False, font=dict(color="#526079", size=12, family=FONT_MONO))]
        )
        return fig

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Fused Affect Dynamics (Valence & Arousal)", "Acoustic Telemetry (Pitch & Energy)"),
    )

    x_vals = list(range(len(history)))

    # Track 1: Valence, Arousal, Confidence
    valences = [h.affect.valence for h in history]
    arousals = [h.affect.arousal for h in history]
    confidences = [h.confidence for h in history]

    fig.add_trace(
        go.Scatter(x=x_vals, y=valences, mode='lines+markers', name='Valence', line=dict(color='#10b981', width=1.75), marker=dict(size=4)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_vals, y=arousals, mode='lines+markers', name='Arousal', line=dict(color='#f59e0b', width=1.75), marker=dict(size=4)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_vals, y=confidences, mode='lines', name='Confidence', line=dict(color='#3b82f6', width=1.2, dash='dot')),
        row=1, col=1
    )

    # Track 2: Acoustic features (F0 pitch, RMS energy, vocal stress)
    pitches = [h.voice.acoustics.pitch_hz if h.voice and h.voice.acoustics else 0.0 for h in history]
    stresses = [h.voice.vocal_stress_level if h.voice else 0.0 for h in history]
    rms_vals = [h.voice.acoustics.rms_energy * 20.0 if h.voice and h.voice.acoustics else 0.0 for h in history]

    fig.add_trace(
        go.Scatter(x=x_vals, y=pitches, mode='lines', name='Pitch F0 (Hz)', line=dict(color='#06b6d4', width=1.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_vals, y=stresses, mode='lines', name='Vocal Stress', line=dict(color='#ef4444', width=1.5, dash='dash')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_vals, y=rms_vals, mode='lines', name='Volume (RMSx20)', line=dict(color='#a855f7', width=1.2)),
        row=2, col=1
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=25, r=25, t=35, b=25),
        height=360,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            font=dict(color=AXIS_COLOR, size=8, family=FONT_MONO)
        ),
    )

    fig.update_xaxes(gridcolor=GRID_COLOR, tickfont=dict(color="#526079", size=8, family=FONT_MONO))
    fig.update_yaxes(gridcolor=GRID_COLOR, tickfont=dict(color="#526079", size=8, family=FONT_MONO))

    return fig


# =====================================================================
# Phase 5: Multi-Subject & Dyadic Interaction Telemetry Charts
# =====================================================================

def render_dyadic_synchrony_chart(
    times: List[float],
    val_a: List[float],
    val_b: List[float],
    label_a: str = "Participant A",
    label_b: str = "Participant B",
    synchrony_score: float = 0.0,
) -> go.Figure:
    """Renders dual synchronized valence waveforms comparing two participants over time."""
    fig = go.Figure()

    t_min = min(times) if times else 0
    t_max = max(times) if times else 10

    # Zero reference line
    fig.add_shape(
        type="line", x0=t_min, x1=t_max,
        y0=0, y1=0, line=dict(color="rgba(255, 255, 255, 0.15)", dash="dash", width=1)
    )

    # Participant A Trace (Blue)
    fig.add_trace(go.Scatter(
        x=times, y=val_a, mode='lines',
        name=label_a,
        line=dict(color='#3b82f6', width=2.5),
        hoverinfo='x+y+name'
    ))

    # Participant B Trace (Emerald)
    fig.add_trace(go.Scatter(
        x=times, y=val_b, mode='lines',
        name=label_b,
        line=dict(color='#10b981', width=2.5),
        hoverinfo='x+y+name'
    ))

    fig.update_layout(
        title=dict(
            text=f"Affective Valence Synchrony (Pearson r: {synchrony_score:+.2f})",
            font=dict(color="#f8fafc", size=12, family=FONT_DISPLAY)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=25, r=25, t=35, b=25),
        height=280,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=AXIS_COLOR, size=9, family=FONT_MONO)
        ),
        yaxis=dict(range=[-1.05, 1.05], title="Valence", gridcolor=GRID_COLOR, tickfont=dict(color="#526079", size=9, family=FONT_MONO)),
        xaxis=dict(title="Time (seconds)", gridcolor=GRID_COLOR, tickfont=dict(color="#526079", size=9, family=FONT_MONO)),
    )
    return fig


def render_conversational_dominance_pie(speaker_durations: Dict[str, float]) -> go.Figure:
    """Renders a sleek donut chart showing speaking floor dominance between participants."""
    labels = list(speaker_durations.keys()) if speaker_durations else ["Silence"]
    values = list(speaker_durations.values()) if speaker_durations else [1.0]
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#a855f7']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors[:len(labels)], line=dict(color='#0b0f19', width=2)),
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        textfont=dict(color='#ffffff', size=11, family=FONT_DISPLAY)
    )])

    fig.update_layout(
        title=dict(text="Conversational Floor Share", font=dict(color="#f8fafc", size=11, family=FONT_DISPLAY)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=35, b=20),
        height=260,
        showlegend=False,
    )
    return fig


def render_rapport_gauge(rapport_score: float, category: str = "Collaborative") -> go.Figure:
    """Renders a precision semicircular gauge for Dyadic Rapport Index."""
    import numpy as np
    score = float(np.clip(rapport_score, 0.0, 100.0))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="/100", font=dict(color="#f8fafc", size=24, family=FONT_DISPLAY)),
        title=dict(text=f"DYADIC RAPPORT: {category.upper()}", font=dict(color=AXIS_COLOR, size=11, family=FONT_MONO)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=GRID_COLOR, tickfont=dict(color=AXIS_COLOR, size=8, family=FONT_MONO)),
            bar=dict(color="#3b82f6", thickness=0.3),
            bgcolor="rgba(255, 255, 255, 0.05)",
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color="rgba(239, 68, 68, 0.25)"),
                dict(range=[40, 70], color="rgba(245, 158, 11, 0.25)"),
                dict(range=[70, 100], color="rgba(16, 185, 129, 0.25)"),
            ],
            threshold=dict(
                line=dict(color="#10b981" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444", width=4),
                thickness=0.8,
                value=score
            )
        )
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=25, r=25, t=35, b=20),
        height=220,
    )
    return fig


def render_turn_taking_timeline(turns: List[Any]) -> go.Figure:
    """Renders horizontal segment timeline of conversational turn taking."""
    fig = go.Figure()
    palette = {'Speaker_0': '#3b82f6', 'Speaker_1': '#10b981', 'Interviewer': '#3b82f6', 'Candidate': '#10b981'}

    for t in turns:
        spk = getattr(t, "speaker_id", t.get("speaker_id", "Speaker_0") if isinstance(t, dict) else "Speaker_0")
        st = getattr(t, "start_time", t.get("start_time", 0.0) if isinstance(t, dict) else 0.0)
        et = getattr(t, "end_time", t.get("end_time", 0.0) if isinstance(t, dict) else 0.0)
        dur = getattr(t, "duration", et - st)
        emo = getattr(t, "dominant_emotion", "neutral")

        color = palette.get(spk, '#f59e0b')
        fig.add_trace(go.Bar(
            x=[dur],
            y=[spk],
            base=[st],
            orientation='h',
            marker=dict(color=color, opacity=0.85, line=dict(color='#0f172a', width=1)),
            name=spk,
            hovertext=f"{spk}: {st:.1f}s - {et:.1f}s ({dur:.1f}s) | {emo}",
            hoverinfo='text',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(text="Conversational Turn-Taking & Floor Transitions", font=dict(color="#f8fafc", size=11, family=FONT_DISPLAY)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=20, r=20, t=35, b=25),
        height=200,
        barmode='stack',
        xaxis=dict(title="Timeline (seconds)", gridcolor=GRID_COLOR, tickfont=dict(color="#526079", size=9, family=FONT_MONO)),
        yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO)),
    )
    return fig
