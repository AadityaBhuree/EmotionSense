"""Phase 13: Precision Plotly Visualizations for Forensic Credibility & Veracity Telemetry.

Provides interactive Credibility & Deception Risk tachometer dials, Micro-Expression
Leakage event timelines, 5-channel Multimodal Polygraph radar spider charts, and
responsive laboratory-grade HTML HUD cards for forensic veracity evaluation.
"""

from typing import List, Optional
import plotly.graph_objects as go

from src.core.credibility_models import (
    MicroLeakageEvent,
    MultimodalPolygraphProfile,
    CredibilitySnapshot,
)

# Visual Theme Constants
CHART_BG = "rgba(18, 22, 34, 0.6)"
GRID_COLOR = "rgba(255, 255, 255, 0.05)"
AXIS_COLOR = "#94a3b8"
FONT_DISPLAY = "Plus Jakarta Sans, -apple-system, sans-serif"
FONT_MONO = "JetBrains Mono, monospace"


def render_credibility_tachometer_gauge(
    snapshot: CredibilitySnapshot,
    height: int = 240,
) -> go.Figure:
    """Renders semicircular tachometer dial showing Credibility Score (0-100%) and Veracity Tier."""
    val = float(snapshot.credibility_score) * 100.0
    risk_val = float(snapshot.deception_risk_index) * 100.0

    if risk_val < 25.0:
        bar_color = "#10b981"  # Emerald Veridical
        tier_label = "VERIDICAL AUTHENTIC"
    elif risk_val < 50.0:
        bar_color = "#38bdf8"  # Sky Blue Cognitive Strain
        tier_label = "COGNITIVE STRAIN"
    elif risk_val < 75.0:
        bar_color = "#f59e0b"  # Amber Suspicious Incongruence
        tier_label = "SUSPICIOUS INCONGRUENCE"
    else:
        bar_color = "#ef4444"  # Crimson High Deception Risk
        tier_label = "HIGH DECEPTION RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'color': bar_color, 'size': 26, 'family': FONT_MONO}},
        title={
            'text': f"<b>CREDIBILITY & VERACITY INDEX</b><br><span style='color:{bar_color}; font-size:11px; font-family:{FONT_MONO}'>[{tier_label}]</span>",
            'font': {'size': 12, 'color': '#f1f5f9', 'family': FONT_DISPLAY}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': AXIS_COLOR, 'tickwidth': 1, 'tickfont': {'color': AXIS_COLOR, 'size': 9, 'family': FONT_MONO}},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': 'rgba(255,255,255,0.04)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(239, 68, 68, 0.20)'},
                {'range': [25, 50], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [50, 75], 'color': 'rgba(56, 189, 248, 0.15)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.18)'},
            ],
            'threshold': {
                'line': {'color': '#10b981', 'width': 3},
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


def render_micro_leakage_timeline(
    events: Optional[List[MicroLeakageEvent]] = None,
    height: int = 240,
) -> go.Figure:
    """Renders temporal scatter timeline of sub-200ms micro-expression leakage bursts."""
    fig = go.Figure()

    if not events:
        # Default placeholder demonstration timeline
        fig.add_trace(go.Scatter(
            x=[1.2, 4.5, 8.1],
            y=[120.0, 150.0, 95.0],
            mode='markers+text',
            marker=dict(
                size=[14, 18, 12],
                color=['#f59e0b', '#ef4444', '#a855f7'],
                symbol='diamond',
                line=dict(color='#ffffff', width=1.5)
            ),
            text=["Fear (120ms)", "Contempt (150ms)", "Anger (95ms)"],
            textposition="top center",
            textfont=dict(color='#f1f5f9', size=9, family=FONT_MONO),
            hovertemplate='Time: %{x}s<br>Duration: %{y}ms<extra></extra>',
            name='Micro-Flashes',
        ))
    else:
        ts = [max(0.1, round(e.timestamp, 1)) for e in events]
        durs = [e.duration_ms for e in events]
        labels = [f"{e.leaked_affect} ({e.duration_ms:.0f}ms)" for e in events]
        colors = ['#ef4444' if "fear" in e.leaked_affect.lower() else '#f59e0b' for e in events]

        fig.add_trace(go.Scatter(
            x=ts,
            y=durs,
            mode='markers+text',
            marker=dict(
                size=16,
                color=colors,
                symbol='diamond',
                line=dict(color='#ffffff', width=1.5)
            ),
            text=labels,
            textposition="top center",
            textfont=dict(color='#f1f5f9', size=9, family=FONT_MONO),
            hovertemplate='Time: %{x}s<br>Duration: %{y}ms<br>Conflicted AUs: %{text}<extra></extra>',
            name='Leakage Events',
        ))

    # Threshold line for sub-200ms boundary
    fig.add_shape(
        type="line",
        x0=0, y0=200.0, x1=1, y1=200.0,
        xref="paper", yref="y",
        line=dict(color="rgba(239, 68, 68, 0.4)", width=1, dash="dash"),
    )

    fig.update_layout(
        title=dict(
            text="<b>MICRO-EXPRESSION LEAKAGE BURSTS (&lt;200ms)</b>",
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
            title=dict(text="Timeline (sec)", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            title=dict(text="Burst Duration (ms)", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
            range=[0, 300],
        ),
        showlegend=False,
    )
    return fig


def render_polygraph_multimodal_stress_radar(
    polygraph: MultimodalPolygraphProfile,
    height: int = 260,
) -> go.Figure:
    """Renders 5-channel polar radar spider chart decomposing polygraphic deception markers."""
    categories = [
        "Facial Incongruence",
        "Voice Stress (VSA)",
        "Pupil Dilation Strain",
        "Pacifying Adaptor",
        "Pulse Surge",
    ]
    vals = [
        float(polygraph.facial_incongruence_score) * 100.0,
        float(polygraph.voice_stress_score) * 100.0,
        float(polygraph.pupil_dilation_strain) * 100.0,
        float(polygraph.pacifying_adaptor_score) * 100.0,
        float(polygraph.autonomic_pulse_surge_score) * 100.0,
    ]
    # Close the polygon loop
    categories.append(categories[0])
    vals.append(vals[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals,
        theta=categories,
        fill='toself',
        fillcolor='rgba(244, 63, 94, 0.20)',
        line=dict(color='#f43f5e', width=2),
        name='Deception Indicators',
        hovertemplate='%{theta}: %{r:.1f}%<extra></extra>',
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=CHART_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color=AXIS_COLOR, size=8, family=FONT_MONO),
                gridcolor=GRID_COLOR,
            ),
            angularaxis=dict(
                tickfont=dict(color='#cbd5e1', size=9, family=FONT_DISPLAY),
                gridcolor=GRID_COLOR,
            ),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=35, r=35, t=35, b=25),
        height=height,
        showlegend=False,
        title=dict(
            text="<b>POLYGRAPHIC MULTI-CHANNEL DECOMPOSITION</b>",
            font=dict(size=12, color='#f1f5f9', family=FONT_DISPLAY),
            x=0.02, y=0.96
        ),
    )
    return fig


def render_credibility_hud_html(snapshot: CredibilitySnapshot) -> str:
    """Generates responsive laboratory-grade HTML HUD card for real-time forensic veracity."""
    cred = snapshot.credibility_score * 100.0
    risk = snapshot.deception_risk_index * 100.0
    fac = snapshot.facial_veracity
    vox = snapshot.voice_stress

    duchenne_congruence = f"{fac.duchenne_congruence * 100.0:.0f}%"
    vsa_stress = f"{vox.stress_index * 100.0:.0f}%"
    cpp_val = f"{vox.cpp_db:.1f} dB"
    leakages_count = str(fac.leakage_events_count)

    if risk < 25.0:
        pill_bg = "rgba(16, 185, 129, 0.15)"
        pill_border = "#10b981"
        pill_text = "#34d399"
        tier_title = "VERIDICAL AUTHENTIC"
    elif risk < 50.0:
        pill_bg = "rgba(56, 189, 248, 0.15)"
        pill_border = "#38bdf8"
        pill_text = "#7dd3fc"
        tier_title = "COGNITIVE STRAIN"
    elif risk < 75.0:
        pill_bg = "rgba(245, 158, 11, 0.15)"
        pill_border = "#f59e0b"
        pill_text = "#fbbf24"
        tier_title = "SUSPICIOUS INCONGRUENCE"
    else:
        pill_bg = "rgba(239, 68, 68, 0.20)"
        pill_border = "#ef4444"
        pill_text = "#f87171"
        tier_title = "HIGH DECEPTION RISK"

    flags_html = "".join([
        f"<span style='background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.3); color:#fca5a5; padding:2px 8px; border-radius:4px; font-size:10px; margin-right:4px;'>{fl}</span>"
        for fl in snapshot.active_flags[:3]
    ]) if snapshot.active_flags else "<span style='color:#94a3b8; font-size:10px;'>No deception flags active</span>"

    return f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px 18px; backdrop-filter: blur(12px); margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; margin-bottom: 12px;">
            <div style="font-family: {FONT_DISPLAY}; font-weight: 600; font-size: 13px; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{pill_border}; box-shadow:0 0 8px {pill_border};"></span>
                FORENSIC VERACITY &amp; CREDIBILITY SENTINEL HUD
            </div>
            <div style="background: {pill_bg}; border: 1px solid {pill_border}; color: {pill_text}; font-family: {FONT_MONO}; font-size: 11px; padding: 2px 10px; border-radius: 20px; font-weight: 600;">
                {tier_title}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; text-align: center;">
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">CREDIBILITY SCORE</div>
                <div style="font-size: 18px; font-weight: 700; color: {pill_text}; font-family: {FONT_MONO};">{cred:.0f}%</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">DUCHENNE SMILE</div>
                <div style="font-size: 18px; font-weight: 700; color: {'#10b981' if fac.duchenne_congruence >= 0.7 else '#f59e0b'}; font-family: {FONT_MONO};">{duchenne_congruence}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">VOICE STRESS (VSA)</div>
                <div style="font-size: 18px; font-weight: 700; color: {'#ef4444' if vox.is_voice_stressed else '#38bdf8'}; font-family: {FONT_MONO};">{vsa_stress}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">HARMONIC CPP</div>
                <div style="font-size: 16px; font-weight: 700; color: #a78bfa; font-family: {FONT_MONO};">{cpp_val}</div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 10px; color: #94a3b8; font-family: {FONT_DISPLAY};">MICRO-FLASHES</div>
                <div style="font-size: 18px; font-weight: 700; color: {'#ef4444' if fac.leakage_events_count > 0 else '#10b981'}; font-family: {FONT_MONO};">{leakages_count}</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-family: {FONT_MONO}; font-size: 11px; color: #94a3b8; display: flex; align-items: center; gap: 6px;">
            <span>ACTIVE ANOMALY FLAGS:</span>
            {flags_html}
        </div>
    </div>
    """
