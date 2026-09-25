"""Phase 12: Precision Plotly Visualizations for Somatosensory Kinematics & Postural Ergonomics.

Provides interactive Postural Ergonomics alignment diagrams, Hand-to-Face Micro-Gesture
Adaptor timeline charts, Kinetic Fidgeting & Restlessness waveforms, semicircular Psychomotor
Agitation gauges, and responsive neuro-instrument HTML HUD banners.
"""

from typing import List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go

from src.core.somatosensory_models import (
    PostureState,
    AdaptorCategory,
    PosturalMetrics,
    MicroGestureAdaptor,
    FidgetingDynamics,
    PsychomotorAgitationIndex,
    SomatosensorySnapshot,
)

# Visual Theme Constants
CHART_BG = "rgba(18, 22, 34, 0.6)"
GRID_COLOR = "rgba(255, 255, 255, 0.05)"
AXIS_COLOR = "#94a3b8"
FONT_DISPLAY = "Plus Jakarta Sans, -apple-system, sans-serif"
FONT_MONO = "JetBrains Mono, monospace"


def render_postural_ergonomics_diagram(
    posture: PosturalMetrics,
    height: int = 260,
) -> go.Figure:
    """Renders 2D schematic of head-neck-shoulder ergonomics and spinal alignment."""
    fig = go.Figure()

    # Base reference vertical spine line
    fig.add_shape(
        type="line", x0=0.5, y0=0.15, x1=0.5, y1=0.85,
        line=dict(color="rgba(255, 255, 255, 0.12)", width=1, dash="dash"),
    )

    # Shoulder line coordinates with elevation tilt
    asym = float(posture.shoulder_elevation_asymmetry)
    left_y = 0.35 + asym * 0.25
    right_y = 0.35 - asym * 0.25

    fig.add_trace(go.Scatter(
        x=[0.25, 0.75],
        y=[left_y, right_y],
        mode='lines+markers',
        line=dict(color='#38bdf8', width=3),
        marker=dict(size=10, color=['#0284c7', '#38bdf8']),
        name='Shoulder Axis',
        hovertemplate='Shoulder Asymmetry: %{y:.2f}<extra></extra>',
    ))

    # Head position (shifted by spinal tilt and slump)
    tilt_rad = np.radians(posture.spinal_tilt_deg)
    head_x = float(0.5 + 0.35 * np.sin(tilt_rad))
    head_y = float(0.75 - posture.slump_index * 0.25)

    # Spine line connecting mid-shoulder to head
    mid_shoulder_y = (left_y + right_y) / 2.0
    spine_color = '#10b981' if posture.posture_state == PostureState.UPRIGHT.value else ('#f59e0b' if posture.posture_state == PostureState.LATERAL_LEAN.value else '#ef4444')

    fig.add_trace(go.Scatter(
        x=[0.5, head_x],
        y=[mid_shoulder_y, head_y],
        mode='lines',
        line=dict(color=spine_color, width=3),
        name='Cervical Spine Vector',
        hoverinfo='skip',
    ))

    # Head marker circle
    fig.add_shape(
        type="circle",
        x0=head_x - 0.08, y0=head_y - 0.08,
        x1=head_x + 0.08, y1=head_y + 0.08,
        line=dict(color=spine_color, width=2),
        fillcolor="rgba(56, 189, 248, 0.15)",
    )

    fig.add_trace(go.Scatter(
        x=[head_x], y=[head_y],
        mode='markers+text',
        marker=dict(size=14, color=spine_color),
        text=[f"{posture.forward_head_angle_deg:.0f}°"],
        textposition="top center",
        textfont=dict(color="#f1f5f9", size=10, family=FONT_MONO),
        name='Head Centroid',
        hovertemplate='Forward Head Angle: %{text}<extra></extra>',
    ))

    state_badge = posture.posture_state.replace('_', ' ').upper()
    fig.update_layout(
        title=dict(
            text=f"<b>POSTURAL ERGONOMICS & SPINAL AXIS</b> &nbsp;<span style='color:{spine_color}; font-size:11px; font-family:{FONT_MONO};'>[{state_badge}]</span>",
            font=dict(size=12, color='#f1f5f9', family=FONT_DISPLAY),
            x=0.02, y=0.96
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=25, r=20, t=35, b=25),
        height=height,
        xaxis=dict(
            range=[0.1, 0.9],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=[0.1, 0.95],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        showlegend=False,
    )
    return fig


def render_adaptor_timeline_chart(
    adaptors: List[MicroGestureAdaptor],
    height: int = 240,
) -> go.Figure:
    """Renders horizontal bar chart of active hand-to-face micro-gesture adaptors."""
    fig = go.Figure()

    if not adaptors:
        # Default placeholder bars for baseline monitoring
        types = ["Chin Support", "Mouth Cover", "Temple Rub", "Neck Touch"]
        distances = [0.85, 0.88, 0.92, 0.95]
        colors = ['rgba(255,255,255,0.08)'] * 4
    else:
        types = [a.adaptor_type.replace('_', ' ') for a in adaptors]
        distances = [1.0 - a.proximity_distance for a in adaptors]  # Higher = closer
        colors = [
            '#ef4444' if a.category == AdaptorCategory.PACIFYING_STRESS.value
            else ('#f59e0b' if a.category == AdaptorCategory.FATIGUE_OVERLOAD.value
                  else ('#a78bfa' if a.category == AdaptorCategory.DEFENSIVE_UNCERTAIN.value else '#38bdf8'))
            for a in adaptors
        ]

    fig.add_trace(go.Bar(
        x=distances,
        y=types,
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        hovertemplate='Proximity Contact: %{x:.2f}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text="<b>HAND-TO-FACE MICRO-GESTURE ADAPTORS</b>",
            font=dict(size=12, color='#f1f5f9', family=FONT_DISPLAY),
            x=0.02, y=0.96
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=90, r=20, t=35, b=25),
        height=height,
        xaxis=dict(
            range=[0.0, 1.0],
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            title=dict(text="Contact Proximity (Norm)", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color='#f1f5f9', size=10, family=FONT_DISPLAY),
        ),
        showlegend=False,
    )
    return fig


def render_fidgeting_waveform(
    fidget_history: Optional[List[Tuple[float, float]]] = None,
    current_fidgeting: Optional[FidgetingDynamics] = None,
    height: int = 220,
) -> go.Figure:
    """Renders temporal kinetic energy and restlessness fidgeting waveform."""
    fig = go.Figure()

    if not fidget_history or len(fidget_history) < 2:
        t = np.linspace(0, 10, 40)
        v = 0.01 + 0.005 * np.sin(t * 1.5) + np.random.randn(40) * 0.002
        fig.add_trace(go.Scatter(
            x=t, y=np.clip(v, 0.001, 0.05),
            mode='lines',
            line=dict(color='#10b981', width=1.5),
            name='Kinetic Energy',
        ))
    else:
        ts = [pt[0] - fidget_history[0][0] for pt in fidget_history]
        vals = [pt[1] for pt in fidget_history]
        fig.add_trace(go.Scatter(
            x=ts, y=vals,
            mode='lines',
            line=dict(color='#f59e0b' if current_fidgeting and current_fidgeting.is_fidgeting else '#38bdf8', width=2),
            fill='tozeroy',
            fillcolor='rgba(56, 189, 248, 0.1)',
            name='Kinetic Displacement',
            hovertemplate='Time: %{x:.1f}s<br>Kinetic Energy: %{y:.4f}<extra></extra>',
        ))

    # Agitation threshold guideline
    fig.add_shape(
        type="line",
        x0=0, y0=0.015, x1=1, y1=0.015,
        xref="paper", yref="y",
        line=dict(color="rgba(239, 68, 68, 0.5)", width=1, dash="dash"),
    )

    restless_val = f"{current_fidgeting.restlessness_score:.2f}" if current_fidgeting else "--"
    fig.update_layout(
        title=dict(
            text=f"<b>KINETIC DISPLACEMENT & FIDGETING ENERGY</b> &nbsp;<span style='color:#38bdf8; font-size:11px; font-family:{FONT_MONO};'>Restlessness: {restless_val}</span>",
            font=dict(size=12, color='#f1f5f9', family=FONT_DISPLAY),
            x=0.02, y=0.96
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=35, r=20, t=35, b=25),
        height=height,
        xaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            title=dict(text="Time Window (sec)", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            title=dict(text="Kinetic Energy", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
        ),
        showlegend=False,
    )
    return fig


def render_psychomotor_agitation_gauge(
    agitation: PsychomotorAgitationIndex,
    height: int = 240,
) -> go.Figure:
    """Renders semicircular tachometer dial showing Psychomotor Agitation Index (0-100%)."""
    val = float(np.clip(agitation.agitation_index, 0.0, 1.0)) * 100.0

    if val < 25.0:
        bar_color = "#10b981"  # Emerald Composed
    elif val < 50.0:
        bar_color = "#38bdf8"  # Cyan Restless Mild
    elif val < 75.0:
        bar_color = "#f59e0b"  # Amber Agitated High
    else:
        bar_color = "#ef4444"  # Crimson Motor Storm

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'color': bar_color, 'size': 26, 'family': FONT_MONO}},
        title={
            'text': f"<b>PSYCHOMOTOR AGITATION INDEX</b><br><span style='color:{bar_color}; font-size:12px; font-family:{FONT_MONO}'>{agitation.tier.replace('_', ' ').upper()}</span>",
            'font': {'size': 12, 'color': '#f1f5f9', 'family': FONT_DISPLAY}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': AXIS_COLOR, 'tickwidth': 1, 'tickfont': {'color': AXIS_COLOR, 'size': 9, 'family': FONT_MONO}},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': 'rgba(255,255,255,0.04)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [25, 50], 'color': 'rgba(56, 189, 248, 0.15)'},
                {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.20)'},
            ],
            'threshold': {
                'line': {'color': '#ef4444', 'width': 3},
                'thickness': 0.75,
                'value': 75.0
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=15),
        height=height,
    )
    return fig


def render_somatosensory_hud_html(snapshot: SomatosensorySnapshot) -> str:
    """Generates responsive neuro-instrument HTML HUD card block for real-time somatosensory telemetry."""
    posture = snapshot.posture
    adaptor = snapshot.primary_adaptor
    fidget = snapshot.fidgeting
    expr = snapshot.expressivity
    agit = snapshot.agitation

    posture_val = posture.posture_state.replace('_', ' ').upper()
    adaptor_val = adaptor.adaptor_type.replace('_', ' ').upper() if adaptor.active else "NONE DETECTED"
    restless_val = f"{fidget.restlessness_score * 100.0:.0f}%"
    express_val = f"{expr.expressivity_score * 100.0:.0f}%"
    slump_val = f"{posture.slump_index * 100.0:.0f}%"

    # Status pill color
    if agit.tier == "Composed":
        pill_bg = "rgba(16, 185, 129, 0.15)"
        pill_border = "#10b981"
        pill_text = "#34d399"
    elif agit.tier == "Restless_Mild":
        pill_bg = "rgba(56, 189, 248, 0.15)"
        pill_border = "#38bdf8"
        pill_text = "#7dd3fc"
    elif agit.tier == "Agitated_High":
        pill_bg = "rgba(245, 158, 11, 0.15)"
        pill_border = "#f59e0b"
        pill_text = "#fbbf24"
    else:
        pill_bg = "rgba(239, 68, 68, 0.2)"
        pill_border = "#ef4444"
        pill_text = "#f87171"

    factors_html = "".join([f"<span style='background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:4px; font-size:10px; margin-right:4px;'>{f}</span>" for f in agit.contributing_factors[:2]])

    retard_badge = ""
    if agit.psychomotor_slowing:
        retard_badge = "<span style='background:rgba(167,139,250,0.25); border:1px solid #a78bfa; color:#c4b5fd; font-size:10px; padding:1px 6px; border-radius:4px; margin-left:6px; font-weight:700;'>PSYCHOMOTOR SLOWING</span>"
    elif fidget.is_fidgeting:
        retard_badge = "<span style='background:rgba(245,158,11,0.2); border:1px solid #f59e0b; color:#fbbf24; font-size:10px; padding:1px 6px; border-radius:4px; margin-left:6px;'>ACTIVE FIDGETING</span>"

    return f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px 18px; backdrop-filter: blur(12px); margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; margin-bottom: 12px;">
            <div style="font-family: {FONT_DISPLAY}; font-weight: 600; font-size: 13px; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10b981; box-shadow:0 0 8px #10b981;"></span>
                SOMATOSENSORY & KINESICS TELEMETRY HUD
                {retard_badge}
            </div>
            <div style="background: {pill_bg}; border: 1px solid {pill_border}; color: {pill_text}; font-family: {FONT_MONO}; font-size: 11px; padding: 2px 10px; border-radius: 20px; font-weight: 600;">
                {agit.tier.replace('_', ' ').upper()}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; text-align: center;">
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">POSTURE ALIGNMENT</div>
                <div style="font-size: 14px; font-weight: 700; color: #38bdf8; font-family: {FONT_MONO};">{posture_val}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">SLUMP INDEX</div>
                <div style="font-size: 18px; font-weight: 700; color: {'#ef4444' if posture.slump_index >= 0.45 else '#10b981'}; font-family: {FONT_MONO};">{slump_val}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">SELF-TOUCH ADAPTOR</div>
                <div style="font-size: 13px; font-weight: 700; color: {'#a78bfa' if adaptor.active else '#94a3b8'}; font-family: {FONT_MONO};">{adaptor_val}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">RESTLESSNESS</div>
                <div style="font-size: 18px; font-weight: 700; color: {'#f59e0b' if fidget.is_fidgeting else '#38bdf8'}; font-family: {FONT_MONO};">{restless_val}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">KINESIC EXPRESSIVITY</div>
                <div style="font-size: 18px; font-weight: 700; color: #34d399; font-family: {FONT_MONO};">{express_val}</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-family: {FONT_MONO}; font-size: 11px; color: #94a3b8; display: flex; align-items: center; gap: 6px;">
            <span>SOMATIC BIOMARKERS:</span>
            {factors_html}
        </div>
    </div>
    """
