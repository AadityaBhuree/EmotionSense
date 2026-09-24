"""Phase 11: Precision Plotly Visualizations for Oculomotor Telemetry & Cognitive Workload.

Provides interactive Gaze Dispersion heatmaps, NASA-TLX 4-dimensional radar dials,
semicircular Cognitive Workload & Mental Overload gauges, and responsive HTML HUD banners.
"""

from typing import List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go

from src.core.cognitive_models import (
    GazeTelemetry,
    NASATLXDimensions,
    CognitiveWorkloadRecord,
    OculomotorSnapshot,
)

# Visual Theme Constants
CHART_BG = "rgba(18, 22, 34, 0.6)"
GRID_COLOR = "rgba(255, 255, 255, 0.05)"
AXIS_COLOR = "#94a3b8"
FONT_DISPLAY = "Plus Jakarta Sans, -apple-system, sans-serif"
FONT_MONO = "JetBrains Mono, monospace"


def render_gaze_dispersion_chart(
    gaze_points: Optional[List[Tuple[float, float]]] = None,
    current_gaze: Optional[GazeTelemetry] = None,
    height: int = 260,
) -> go.Figure:
    """Renders 2D screen focal coordinate scatter and gaze fixation dispersion heatmap."""
    fig = go.Figure()

    # Screen bounding zone
    fig.add_shape(
        type="rect",
        x0=0.0, y0=0.0, x1=1.0, y1=1.0,
        line=dict(color="rgba(255, 255, 255, 0.15)", width=1, dash="dot"),
        fillcolor="rgba(255, 255, 255, 0.02)",
    )

    # Center target crosshairs
    fig.add_shape(type="line", x0=0.5, y0=0.1, x1=0.5, y1=0.9, line=dict(color="rgba(255,255,255,0.06)", width=1))
    fig.add_shape(type="line", x0=0.1, y0=0.5, x1=0.9, y1=0.5, line=dict(color="rgba(255,255,255,0.06)", width=1))

    if not gaze_points or len(gaze_points) < 3:
        # Placeholder central dispersion
        t = np.linspace(0, 2 * np.pi, 20)
        x_dummy = 0.5 + 0.06 * np.cos(t) + np.random.randn(20) * 0.02
        y_dummy = 0.5 + 0.06 * np.sin(t) + np.random.randn(20) * 0.02
        fig.add_trace(go.Scatter(
            x=x_dummy, y=y_dummy,
            mode='markers',
            marker=dict(size=5, color='rgba(56, 189, 248, 0.3)'),
            name='Calibrating Gaze Track...',
        ))
    else:
        pts = np.asarray(gaze_points, dtype=np.float64)
        xs = np.clip(pts[:, 0], 0.02, 0.98)
        ys = np.clip(pts[:, 1], 0.02, 0.98)

        # Gaze trail points
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers',
            marker=dict(
                size=6,
                color=np.linspace(0.2, 1.0, len(xs)),
                colorscale=[[0, 'rgba(56, 189, 248, 0.15)'], [1, '#38bdf8']],
                showscale=False,
            ),
            name='Gaze Trail',
            hovertemplate='Focal X: %{x:.2f}<br>Focal Y: %{y:.2f}<extra></extra>',
        ))

        # Dispersion boundary ellipse around mean focal centroid
        mean_x, mean_y = float(np.mean(xs)), float(np.mean(ys))
        std_x, std_y = float(np.std(xs)), float(np.std(ys))
        fig.add_shape(
            type="circle",
            x0=max(0.0, mean_x - std_x * 1.5),
            y0=max(0.0, mean_y - std_y * 1.5),
            x1=min(1.0, mean_x + std_x * 1.5),
            y1=min(1.0, mean_y + std_y * 1.5),
            line=dict(color="rgba(245, 158, 11, 0.4)", width=1.5, dash="dash"),
            fillcolor="rgba(245, 158, 11, 0.06)",
        )

    # Current focal gaze marker
    cur_x = current_gaze.screen_x if current_gaze else 0.5
    cur_y = current_gaze.screen_y if current_gaze else 0.5
    fig.add_trace(go.Scatter(
        x=[cur_x], y=[cur_y],
        mode='markers',
        marker=dict(size=12, color='#10b981', symbol='cross', line=dict(color='#ffffff', width=1.5)),
        name='Active Fixation',
        hoverinfo='skip',
    ))

    disp_val = f"{current_gaze.dispersion_area:.2f}" if current_gaze else "--"
    fig.update_layout(
        title=dict(
            text=f"<b>OCULOMOTOR GAZE DISPERSION</b> &nbsp;<span style='color:#38bdf8; font-size:11px; font-family:{FONT_MONO};'>Tunneling: {disp_val}</span>",
            font=dict(size=12, color='#f1f5f9', family=FONT_DISPLAY),
            x=0.02, y=0.96
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=25, r=20, t=35, b=25),
        height=height,
        xaxis=dict(
            range=[0.0, 1.0],
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            title=dict(text="Screen Horizontal (X)", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
        ),
        yaxis=dict(
            range=[0.0, 1.0],
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            title=dict(text="Screen Vertical (Y)", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
        ),
        showlegend=False,
    )
    return fig


def render_nasa_tlx_radar(
    nasa_tlx: NASATLXDimensions,
    height: int = 250,
) -> go.Figure:
    """Renders 4-axis polar radar chart for NASA-TLX dimensional workload profiling."""
    categories = ['Mental Demand', 'Temporal Demand', 'Effort', 'Frustration']
    values = [
        float(nasa_tlx.mental_demand),
        float(nasa_tlx.temporal_demand),
        float(nasa_tlx.effort),
        float(nasa_tlx.frustration),
    ]
    # Close polygon
    cat_closed = categories + [categories[0]]
    val_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=val_closed,
        theta=cat_closed,
        fill='toself',
        fillcolor='rgba(14, 165, 233, 0.22)',
        line=dict(color='#0ea5e9', width=2),
        name='NASA-TLX Subscales',
        hoverinfo='r+theta',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=8, color=AXIS_COLOR, family=FONT_MONO),
                gridcolor=GRID_COLOR,
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color='#f1f5f9', family=FONT_DISPLAY),
                gridcolor=GRID_COLOR,
            ),
            bgcolor=CHART_BG,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=35, r=35, t=35, b=25),
        height=height,
        title=dict(
            text="<b>NASA-TLX WORKLOAD RADAR</b>",
            font=dict(size=12, color='#f1f5f9', family=FONT_DISPLAY),
            x=0.02, y=0.96
        ),
        showlegend=False,
    )
    return fig


def render_cognitive_workload_gauge(
    workload: CognitiveWorkloadRecord,
    height: int = 240,
) -> go.Figure:
    """Renders semicircular tachometer dial showing unified Cognitive Workload Index (0-100%)."""
    val = float(np.clip(workload.workload_index, 0.0, 1.0)) * 100.0

    if val < 25.0:
        bar_color = "#10b981"  # Emerald Low
    elif val < 60.0:
        bar_color = "#38bdf8"  # Cyan Optimal
    elif val < 80.0:
        bar_color = "#f59e0b"  # Amber High Effort
    else:
        bar_color = "#ef4444"  # Crimson Cognitive Overload

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'color': bar_color, 'size': 26, 'family': FONT_MONO}},
        title={
            'text': f"<b>COGNITIVE WORKLOAD INDEX</b><br><span style='color:{bar_color}; font-size:12px; font-family:{FONT_MONO}'>{workload.tier.replace('_', ' ').upper()}</span>",
            'font': {'size': 12, 'color': '#f1f5f9', 'family': FONT_DISPLAY}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': AXIS_COLOR, 'tickwidth': 1, 'tickfont': {'color': AXIS_COLOR, 'size': 9, 'family': FONT_MONO}},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': 'rgba(255,255,255,0.04)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [25, 60], 'color': 'rgba(56, 189, 248, 0.15)'},
                {'range': [60, 80], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.20)'},
            ],
            'threshold': {
                'line': {'color': '#ef4444', 'width': 3},
                'thickness': 0.75,
                'value': 80.0
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


def render_oculomotor_hud_html(snapshot: OculomotorSnapshot) -> str:
    """Generates responsive neuro-instrument HTML HUD card block for real-time oculometrics."""
    pupil = snapshot.pupillometry
    blinks = snapshot.blinks
    gaze = snapshot.gaze
    wl = snapshot.workload

    pir_val = f"{pupil.pupil_diameter_ratio:.3f}"
    cpr_val = f"+{pupil.dilation_change_pct:.1f}%" if pupil.dilation_change_pct >= 0 else f"{pupil.dilation_change_pct:.1f}%"
    blink_rate_val = f"{blinks.blink_rate_bpm:.1f} BPM"
    perclos_val = f"{blinks.perclos * 100.0:.1f}%"
    fix_val = f"{gaze.fixation_duration_ms:.0f} ms"
    sacc_val = f"{gaze.saccade_velocity_deg_s:.0f} °/s"

    # Status pill color
    if wl.tier == "Low_Load":
        pill_bg = "rgba(16, 185, 129, 0.15)"
        pill_border = "#10b981"
        pill_text = "#34d399"
    elif wl.tier == "Optimal_Engagement":
        pill_bg = "rgba(56, 189, 248, 0.15)"
        pill_border = "#38bdf8"
        pill_text = "#7dd3fc"
    elif wl.tier == "High_Effort":
        pill_bg = "rgba(245, 158, 11, 0.15)"
        pill_border = "#f59e0b"
        pill_text = "#fbbf24"
    else:
        pill_bg = "rgba(239, 68, 68, 0.2)"
        pill_border = "#ef4444"
        pill_text = "#f87171"

    factors_html = "".join([f"<span style='background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:4px; font-size:10px; margin-right:4px;'>{f}</span>" for f in wl.contributing_factors[:2]])

    drowsy_badge = ""
    if blinks.micro_sleep_detected:
        drowsy_badge = "<span style='background:rgba(239,68,68,0.25); border:1px solid #ef4444; color:#f87171; font-size:10px; padding:1px 6px; border-radius:4px; margin-left:6px; font-weight:700;'>MICRO-SLEEP</span>"
    elif blinks.blink_suppressed:
        drowsy_badge = "<span style='background:rgba(56,189,248,0.2); border:1px solid #38bdf8; color:#7dd3fc; font-size:10px; padding:1px 6px; border-radius:4px; margin-left:6px;'>BLINK SUPPRESSION</span>"

    return f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px 18px; backdrop-filter: blur(12px); margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; margin-bottom: 12px;">
            <div style="font-family: {FONT_DISPLAY}; font-weight: 600; font-size: 13px; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#38bdf8; box-shadow:0 0 8px #38bdf8;"></span>
                OCULOMOTOR & COGNITIVE WORKLOAD HUD
                {drowsy_badge}
            </div>
            <div style="background: {pill_bg}; border: 1px solid {pill_border}; color: {pill_text}; font-family: {FONT_MONO}; font-size: 11px; padding: 2px 10px; border-radius: 20px; font-weight: 600;">
                {wl.tier.replace('_', ' ').upper()}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; text-align: center;">
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">PUPIL RATIO (PIR)</div>
                <div style="font-size: 18px; font-weight: 700; color: #38bdf8; font-family: {FONT_MONO};">{pir_val} <span style="font-size:10px; font-weight:400; color:#94a3b8;">({cpr_val})</span></div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">BLINK RATE</div>
                <div style="font-size: 18px; font-weight: 700; color: #10b981; font-family: {FONT_MONO};">{blink_rate_val}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">PERCLOS (DROWSY)</div>
                <div style="font-size: 18px; font-weight: 700; color: {'#ef4444' if blinks.perclos > 0.15 else '#f59e0b'}; font-family: {FONT_MONO};">{perclos_val}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">FIXATION DWELL</div>
                <div style="font-size: 18px; font-weight: 700; color: #a78bfa; font-family: {FONT_MONO};">{fix_val}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">SACCADE VELOCITY</div>
                <div style="font-size: 18px; font-weight: 700; color: #e2e8f0; font-family: {FONT_MONO};">{sacc_val}</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-family: {FONT_MONO}; font-size: 11px; color: #94a3b8; display: flex; align-items: center; gap: 6px;">
            <span>NEUROMETRIC FACTORS:</span>
            {factors_html}
        </div>
    </div>
    """
