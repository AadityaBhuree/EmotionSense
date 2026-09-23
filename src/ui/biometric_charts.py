"""Phase 10: Precision Plotly Visualizations for Remote Biometrics & Autonomic Telemetry.

Provides interactive cardiac photoplethysmogram (BVP) waveforms, HRV Poincaré plots,
semicircular autonomic stress gauges, and sympathetic/parasympathetic tone balance bars.
"""

from typing import List, Optional
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

from src.core.biometric_models import (
    PulseMeasurement,
    AutonomicStressRecord,
    BiometricTelemetry,
)

# Visual Theme Constants
CHART_BG = "rgba(18, 22, 34, 0.6)"
GRID_COLOR = "rgba(255, 255, 255, 0.05)"
AXIS_COLOR = "#94a3b8"
FONT_DISPLAY = "Plus Jakarta Sans, -apple-system, sans-serif"
FONT_MONO = "JetBrains Mono, monospace"


def render_bvp_waveform_chart(
    bvp_history: List[float],
    pulse: Optional[PulseMeasurement] = None,
    fps: float = 30.0,
    height: int = 240,
) -> go.Figure:
    """Renders interactive Blood Volume Pulse (BVP) photoplethysmogram with systolic peaks."""
    fig = go.Figure()

    if not bvp_history or len(bvp_history) < 5:
        # Placeholder waveform
        x_dummy = np.linspace(0, 4, 120)
        y_dummy = 0.5 * np.sin(2 * np.pi * 1.2 * x_dummy) + 0.15 * np.sin(4 * np.pi * 1.2 * x_dummy)
        fig.add_trace(go.Scatter(
            x=x_dummy,
            y=y_dummy,
            mode='lines',
            line=dict(color='rgba(16, 185, 129, 0.35)', width=2, dash='dot'),
            name='Calibrating Optical Signal...',
        ))
        fig.add_annotation(
            text="ACQUIRING CAPILLARY PULSE STREAM",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="#10b981", size=11, family=FONT_MONO)
        )
    else:
        sig = np.array(bvp_history, dtype=np.float64)
        t = np.arange(len(sig)) / fps

        # Main BVP Waveform (neon emerald / cyan glow)
        fig.add_trace(go.Scatter(
            x=t,
            y=sig,
            mode='lines',
            line=dict(color='#10b981', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.08)',
            name='BVP Pulse Wave',
            hoverinfo='x+y'
        ))

        # Detect systolic peaks for visual annotation
        peaks, _ = find_peaks(sig, distance=max(2, int(fps * 0.35)), prominence=np.std(sig) * 0.3)
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=t[peaks],
                y=sig[peaks],
                mode='markers',
                marker=dict(size=8, color='#34d399', symbol='diamond', line=dict(color='#ffffff', width=1)),
                name='Systolic Peak',
                hoverinfo='skip'
            ))

    bpm_text = f"{pulse.bpm:.1f} BPM" if pulse else "-- BPM"
    snr_text = f"SNR: {pulse.signal_quality_snr:.1f} dB" if pulse else "SNR: -- dB"

    fig.update_layout(
        title=dict(
            text=f"<b>OPTICAL PHOTOPLETHYSMOGRAM (BVP)</b> &nbsp;&nbsp;<span style='color:#10b981; font-size:12px; font-family:{FONT_MONO};'>{bpm_text} | {snr_text}</span>",
            font=dict(size=12, color='#f1f5f9', family=FONT_DISPLAY),
            x=0.02, y=0.96
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=35, r=20, t=35, b=25),
        height=height,
        xaxis=dict(
            title=dict(text="Time (seconds)", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="Norm BVP", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.1)',
        ),
        showlegend=False,
    )
    return fig


def render_poincare_plot(
    rr_intervals_ms: List[float],
    height: int = 250,
) -> go.Figure:
    """Renders Poincaré scatter plot (RR_n vs RR_{n+1}) with autonomic variance ellipse."""
    fig = go.Figure()

    if not rr_intervals_ms or len(rr_intervals_ms) < 3:
        # Fallback grid
        fig.add_annotation(
            text="AWAITING SUFFICIENT RR INTERVALS (N >= 3)",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=AXIS_COLOR, size=10, family=FONT_MONO)
        )
        x_min, x_max = 600, 1000
    else:
        rr = np.array(rr_intervals_ms)
        x = rr[:-1]
        y = rr[1:]

        # Identity diagonal line (y = x)
        axis_min = max(400, float(min(np.min(x), np.min(y)) - 50))
        axis_max = min(1300, float(max(np.max(x), np.max(y)) + 50))
        fig.add_trace(go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.15)', dash='dot', width=1),
            hoverinfo='skip',
            name='Identity (y=x)'
        ))

        # Scatter points
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=7,
                color='#38bdf8',
                opacity=0.8,
                line=dict(color='rgba(255, 255, 255, 0.4)', width=0.8)
            ),
            name='RR Pairs',
            hovertemplate='RR_n: %{x:.0f}ms<br>RR_n+1: %{y:.0f}ms<extra></extra>'
        ))

        # Estimate SD1 and SD2
        diffs = y - x
        sd1 = np.std(diffs) / np.sqrt(2.0)
        center_x = np.mean(x)
        center_y = np.mean(y)

        # Draw autonomic tolerance circle
        fig.add_shape(
            type="circle",
            x0=center_x - sd1 * 1.5,
            y0=center_y - sd1 * 1.5,
            x1=center_x + sd1 * 1.5,
            y1=center_y + sd1 * 1.5,
            line=dict(color="rgba(56, 189, 248, 0.4)", width=1.5, dash="dash"),
            fillcolor="rgba(56, 189, 248, 0.05)"
        )

        x_min, x_max = axis_min, axis_max

    fig.update_layout(
        title=dict(
            text="<b>HRV POINCARÉ SCATTER</b> &nbsp;<span style='color:#38bdf8; font-size:11px; font-family:" + FONT_MONO + ";'>RR[n] vs RR[n+1]</span>",
            font=dict(size=12, color='#f1f5f9', family=FONT_DISPLAY),
            x=0.02, y=0.96
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=CHART_BG,
        margin=dict(l=35, r=20, t=35, b=25),
        height=height,
        xaxis=dict(
            title=dict(text="RR[n] (ms)", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            range=[x_min, x_max],
        ),
        yaxis=dict(
            title=dict(text="RR[n+1] (ms)", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            range=[x_min, x_max],
        ),
        showlegend=False,
    )
    return fig


def render_autonomic_stress_gauge(
    stress: AutonomicStressRecord,
    height: int = 240,
) -> go.Figure:
    """Renders semicircular tachometer dial showing Autonomic Stress Index (0.0 to 1.0)."""
    val = float(np.clip(stress.stress_index, 0.0, 1.0)) * 100.0

    # Color threshold based on classification
    if val < 25.0:
        bar_color = "#10b981"  # Emerald Relaxed
    elif val < 52.0:
        bar_color = "#38bdf8"  # Cyan Alert
    elif val < 76.0:
        bar_color = "#f59e0b"  # Amber Elevated
    else:
        bar_color = "#ef4444"  # Crimson Acute

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'color': bar_color, 'size': 26, 'family': FONT_MONO}},
        title={
            'text': f"<b>AUTONOMIC STRESS INDEX</b><br><span style='color:{bar_color}; font-size:12px; font-family:{FONT_MONO}'>{stress.classification.upper()}</span>",
            'font': {'size': 12, 'color': '#f1f5f9', 'family': FONT_DISPLAY}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': AXIS_COLOR, 'tickwidth': 1, 'tickfont': {'color': AXIS_COLOR, 'size': 9, 'family': FONT_MONO}},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': 'rgba(255,255,255,0.04)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [25, 52], 'color': 'rgba(56, 189, 248, 0.15)'},
                {'range': [52, 76], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [76, 100], 'color': 'rgba(239, 68, 68, 0.18)'},
            ],
            'threshold': {
                'line': {'color': '#ef4444', 'width': 3},
                'thickness': 0.75,
                'value': 76.0
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


def render_autonomic_balance_bar(
    stress: AutonomicStressRecord,
    height: int = 100,
) -> go.Figure:
    """Renders horizontal stacked bar comparing Sympathetic vs Parasympathetic autonomic tone."""
    symp = round(stress.sympathetic_tone * 100.0, 1)
    para = round(stress.parasympathetic_tone * 100.0, 1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['Tone'],
        x=[symp],
        name=f'Sympathetic (Strain): {symp}%',
        orientation='h',
        marker=dict(color='#f97316', line=dict(color='rgba(255,255,255,0.2)', width=1)),
        hoverinfo='name',
    ))
    fig.add_trace(go.Bar(
        y=['Tone'],
        x=[para],
        name=f'Parasympathetic (Vagal): {para}%',
        orientation='h',
        marker=dict(color='#10b981', line=dict(color='rgba(255,255,255,0.2)', width=1)),
        hoverinfo='name',
    ))

    fig.update_layout(
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=25),
        height=height,
        xaxis=dict(
            range=[0, 100],
            showgrid=False,
            showticklabels=True,
            ticksuffix='%',
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
        ),
        yaxis=dict(showticklabels=False),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(color='#e2e8f0', size=10, family=FONT_DISPLAY),
        ),
    )
    return fig


def render_biometric_telemetry_hud_html(telemetry: BiometricTelemetry) -> str:
    """Generates a responsive neuro-instrument HTML HUD card block for real-time biometrics."""
    pulse = telemetry.pulse
    hrv = telemetry.hrv
    resp = telemetry.respiration
    stress = telemetry.autonomic_stress

    bpm_val = f"{pulse.bpm:.1f}" if pulse.is_valid else "--"
    rmssd_val = f"{hrv.rmssd_ms:.1f} ms"
    rpm_val = f"{resp.rpm:.1f} RPM"
    si_val = f"{hrv.baevsky_stress_index:.0f}"
    snr_val = f"{pulse.signal_quality_snr:.1f} dB"

    # Status pill color
    if stress.classification == "Relaxed":
        pill_bg = "rgba(16, 185, 129, 0.15)"
        pill_border = "#10b981"
        pill_text = "#34d399"
    elif stress.classification == "Optimal_Alertness":
        pill_bg = "rgba(56, 189, 248, 0.15)"
        pill_border = "#38bdf8"
        pill_text = "#7dd3fc"
    elif stress.classification == "Elevated_Strain":
        pill_bg = "rgba(245, 158, 11, 0.15)"
        pill_border = "#f59e0b"
        pill_text = "#fbbf24"
    else:
        pill_bg = "rgba(239, 68, 68, 0.2)"
        pill_border = "#ef4444"
        pill_text = "#f87171"

    factors_html = "".join([f"<span style='background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:4px; font-size:10px; margin-right:4px;'>{f}</span>" for f in stress.contributing_factors[:2]])

    return f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px 18px; backdrop-filter: blur(12px); margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; margin-bottom: 12px;">
            <div style="font-family: {FONT_DISPLAY}; font-weight: 600; font-size: 13px; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10b981; box-shadow:0 0 8px #10b981;"></span>
                AUTONOMIC BIOMETRICS & rPPG HUD
            </div>
            <div style="background: {pill_bg}; border: 1px solid {pill_border}; color: {pill_text}; font-family: {FONT_MONO}; font-size: 11px; padding: 2px 10px; border-radius: 20px; font-weight: 600;">
                {stress.classification.upper()}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; text-align: center;">
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">PULSE RATE</div>
                <div style="font-size: 18px; font-weight: 700; color: #10b981; font-family: {FONT_MONO};">{bpm_val} <span style="font-size:10px; font-weight:400;">BPM</span></div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">HRV (RMSSD)</div>
                <div style="font-size: 18px; font-weight: 700; color: #38bdf8; font-family: {FONT_MONO};">{rmssd_val}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">RESPIRATION</div>
                <div style="font-size: 18px; font-weight: 700; color: #a78bfa; font-family: {FONT_MONO};">{rpm_val}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">BAEVSKY SI</div>
                <div style="font-size: 18px; font-weight: 700; color: #f59e0b; font-family: {FONT_MONO};">{si_val}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">SIGNAL SNR</div>
                <div style="font-size: 18px; font-weight: 700; color: #e2e8f0; font-family: {FONT_MONO};">{snr_val}</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-family: {FONT_MONO}; font-size: 11px; color: #94a3b8; display: flex; align-items: center; gap: 6px;">
            <span>CLINICAL MARKERS:</span>
            {factors_html}
        </div>
    </div>
    """
