"""Phase 14: Precision Plotly Visualizations for Affective Horizon Forecasting & MDP Dynamics.

Provides interactive Horizon Trajectory Fan charts, Markov Transition Probability heatmaps,
Hazard tachometer dials, Policy simulation comparison bars, and responsive
laboratory-grade HTML HUD cards for predictive emotional sentinel monitoring.
"""

from typing import Dict, List
import plotly.graph_objects as go

from src.core.forecasting_models import (
    HorizonStepForecast,
    PolicyEvaluationResult,
    HorizonForecastSnapshot,
    HorizonRiskTier,
)

# Visual Theme Constants
CHART_BG = "rgba(18, 22, 34, 0.6)"
GRID_COLOR = "rgba(255, 255, 255, 0.05)"
AXIS_COLOR = "#94a3b8"
FONT_DISPLAY = "Plus Jakarta Sans, -apple-system, sans-serif"
FONT_MONO = "JetBrains Mono, monospace"


def render_horizon_trajectory_fan_chart(
    trajectory: List[HorizonStepForecast],
    current_valence: float = 0.0,
    current_arousal: float = 0.0,
    height: int = 260,
) -> go.Figure:
    """Renders multi-step forward Valence and Arousal trajectory with uncertainty fan."""
    fig = go.Figure()

    if not trajectory:
        fig.add_annotation(
            text="No Horizon Trajectory Data Available",
            showarrow=False,
            font={"color": AXIS_COLOR, "size": 13, "family": FONT_MONO},
        )
        fig.update_layout(
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            height=height,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        return fig

    # Build sequence starting from step 0 (current)
    time_labels = ["0s (Now)"] + [f"+{s.time_offset_sec:.0f}s" for s in trajectory]
    valences = [current_valence] + [s.predicted_valence for s in trajectory]
    arousals = [current_arousal] + [s.predicted_arousal for s in trajectory]
    entropies = [0.0] + [s.entropy for s in trajectory]

    # Upper and lower uncertainty bounds around valence
    val_upper = [min(1.0, v + (e * 0.15)) for v, e in zip(valences, entropies)]
    val_lower = [max(-1.0, v - (e * 0.15)) for v, e in zip(valences, entropies)]

    # Uncertainty Fan Fill
    fig.add_trace(go.Scatter(
        x=time_labels + time_labels[::-1],
        y=val_upper + val_lower[::-1],
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name="Valence Confidence Fan",
    ))

    # Valence Trajectory Line
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=valences,
        mode='lines+markers',
        name='Projected Valence',
        line=dict(color='#38bdf8', width=2.5),
        marker=dict(size=7, color='#0284c7', line=dict(color='#e0f2fe', width=1.5)),
        hovertemplate="<b>Step: %{x}</b><br>Valence: %{y:+.2f}<extra></extra>",
    ))

    # Arousal Trajectory Line
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=arousals,
        mode='lines+markers',
        name='Projected Arousal',
        line=dict(color='#f59e0b', width=2, dash='dot'),
        marker=dict(size=6, color='#d97706'),
        hovertemplate="<b>Step: %{x}</b><br>Arousal: %{y:.2f}<extra></extra>",
    ))

    # Danger Hazard Threshold Line
    fig.add_hline(
        y=-0.5,
        line=dict(color="rgba(239, 68, 68, 0.5)", width=1, dash="dash"),
        annotation_text="Escalation Threshold (-0.50)",
        annotation_position="bottom right",
        annotation_font=dict(size=9, color="#f87171", family=FONT_MONO),
    )

    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        height=height,
        margin=dict(l=40, r=20, t=30, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY),
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=10, family=FONT_MONO),
            title=dict(text="Time Horizon Projection", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
        ),
        yaxis=dict(
            range=[-1.05, 1.05],
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=10, family=FONT_MONO),
            title=dict(text="Affective Coordinate", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
        ),
    )

    return fig


def render_markov_transition_heatmap(
    transition_matrix: Dict[str, Dict[str, float]],
    height: int = 300,
) -> go.Figure:
    """Renders 2D Heatmap matrix of Markov state transition probabilities."""
    if not transition_matrix:
        fig = go.Figure()
        fig.add_annotation(text="No Transition Matrix Data", showarrow=False, font={"color": AXIS_COLOR, "size": 12})
        fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=height)
        return fig

    states = list(transition_matrix.keys())
    # Format labels for clean display
    labels = [s.replace("_", " ") for s in states]

    z_vals = []
    for src in states:
        row = [transition_matrix[src].get(dst, 0.0) for dst in states]
        z_vals.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=labels,
        y=labels,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(text="P(Trans)", font=dict(color=AXIS_COLOR, size=10, family=FONT_MONO)),
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            thickness=12,
        ),
        text=[[f"{v:.2f}" for v in row] for row in z_vals],
        texttemplate="%{text}",
        textfont=dict(family=FONT_MONO, size=9, color="#ffffff"),
        hoverongaps=False,
    ))

    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        height=height,
        margin=dict(l=100, r=20, t=30, b=80),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            title=dict(text="Destination State (t+1)", font=dict(color=AXIS_COLOR, size=11, family=FONT_DISPLAY)),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            title=dict(text="Source State (t)", font=dict(color=AXIS_COLOR, size=11, family=FONT_DISPLAY)),
        ),
    )

    return fig


def render_hazard_tachometer_gauge(
    evi: float,
    bch: float,
    risk_tier: str,
    height: int = 230,
) -> go.Figure:
    """Renders semicircular dual-hazard gauge displaying Escalation Velocity and Burnout Crash Risk."""
    max_hazard = max(evi, bch) * 100.0

    if max_hazard < 25.0:
        bar_color = "#10b981"
        tier_label = "STABLE EQUILIBRIUM"
    elif max_hazard < 50.0:
        bar_color = "#38bdf8"
        tier_label = "ELEVATED DRIFT"
    elif max_hazard < 75.0:
        bar_color = "#f59e0b"
        tier_label = "HIGH HAZARD IMPENDING"
    else:
        bar_color = "#ef4444"
        tier_label = "CRITICAL COLLAPSE RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max_hazard,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'color': bar_color, 'size': 26, 'family': FONT_MONO}},
        title={
            'text': f"<b>HORIZON HAZARD INDEX</b><br><span style='color:{bar_color}; font-size:10px; font-family:{FONT_MONO}'>[{tier_label}]</span>",
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
                'value': 75.0,
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        height=height,
        margin=dict(l=25, r=25, t=40, b=15),
    )

    return fig


def render_policy_simulation_comparison_chart(
    policies: List[PolicyEvaluationResult],
    height: int = 240,
) -> go.Figure:
    """Renders horizontal bar chart comparing counterfactual intervention policy impacts."""
    if not policies:
        fig = go.Figure()
        fig.add_annotation(text="No Policy Data Available", showarrow=False, font={"color": AXIS_COLOR, "size": 12})
        fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=height)
        return fig

    labels = [p.action_label for p in policies]
    stability = [p.stability_score * 100.0 for p in policies]
    esc_red = [p.escalation_reduction_percent for p in policies]
    burn_red = [p.burnout_mitigation_percent for p in policies]

    fig = go.Figure()

    # Stability score bar
    fig.add_trace(go.Bar(
        y=labels,
        x=stability,
        orientation='h',
        name='Stability Score (%)',
        marker=dict(color='#38bdf8'),
        hovertemplate="%{y}<br>Stability: %{x:.1f}%<extra></extra>",
    ))

    # Escalation reduction bar
    fig.add_trace(go.Bar(
        y=labels,
        x=esc_red,
        orientation='h',
        name='Escalation Reduction (%)',
        marker=dict(color='#10b981'),
        hovertemplate="%{y}<br>Escalation Reduction: %{x:.1f}%<extra></extra>",
    ))

    # Burnout mitigation bar
    fig.add_trace(go.Bar(
        y=labels,
        x=burn_red,
        orientation='h',
        name='Burnout Mitigation (%)',
        marker=dict(color='#a78bfa'),
        hovertemplate="%{y}<br>Burnout Mitigation: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        barmode='group',
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        height=height,
        margin=dict(l=140, r=20, t=30, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(color=AXIS_COLOR, size=9, family=FONT_DISPLAY),
        ),
        xaxis=dict(
            range=[0, 105],
            gridcolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR, size=9, family=FONT_MONO),
            title=dict(text="Impact Percentage (%)", font=dict(color=AXIS_COLOR, size=10, family=FONT_DISPLAY)),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#f8fafc", size=9, family=FONT_DISPLAY),
        ),
    )

    return fig


def render_forecasting_hud_html(snapshot: HorizonForecastSnapshot) -> str:
    """Generates laboratory-grade responsive HTML HUD card for affective forecasting."""
    evi_pct = int(snapshot.escalation_velocity_index * 100)
    bch_pct = int(snapshot.burnout_crash_hazard * 100)

    # Color scheme for risk tier
    if snapshot.risk_tier == HorizonRiskTier.STABLE_EQUILIBRIUM.value:
        tier_color = "#10b981"
        tier_bg = "rgba(16, 185, 129, 0.12)"
    elif snapshot.risk_tier == HorizonRiskTier.ELEVATED_DRIFT.value:
        tier_color = "#38bdf8"
        tier_bg = "rgba(56, 189, 248, 0.12)"
    elif snapshot.risk_tier == HorizonRiskTier.HIGH_HAZARD_IMPENDING.value:
        tier_color = "#f59e0b"
        tier_bg = "rgba(245, 158, 11, 0.12)"
    else:
        tier_color = "#ef4444"
        tier_bg = "rgba(239, 68, 68, 0.15)"

    rec_action_label = snapshot.optimal_action.replace("_", " ").title()

    html = f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85)); border: 1px solid rgba(56, 189, 248, 0.25); border-left: 4px solid {tier_color}; border-radius: 10px; padding: 12px 18px; margin-bottom: 0.85rem; font-family: 'Plus Jakarta Sans', sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.4rem;">🔮</span>
                <div>
                    <div style="font-weight: 700; color: #f8fafc; font-size: 0.92rem; display: flex; align-items: center; gap: 8px;">
                        <span>Markov Horizon Sentinel & Transition Dynamics</span>
                        <span style="background: {tier_bg}; color: {tier_color}; border: 1px solid {tier_color}; border-radius: 4px; padding: 1px 7px; font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; font-weight: 600;">
                            {snapshot.risk_tier.replace('_', ' ').upper()}
                        </span>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">
                        CURRENT STATE: <b style="color: #38bdf8;">{snapshot.current_state.replace('_', ' ')}</b> • HORIZON: <b>+{snapshot.horizon_steps * 3}s (5 Steps)</b>
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 16px; font-family: 'JetBrains Mono', monospace;">
                <div style="text-align: right;">
                    <div style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase;">Escalation Velocity (EVI)</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: {'#ef4444' if evi_pct > 50 else ('#f59e0b' if evi_pct > 25 else '#10b981')};">{evi_pct}%</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase;">Burnout Crash Hazard</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: {'#ef4444' if bch_pct > 50 else ('#f59e0b' if bch_pct > 25 else '#38bdf8')};">{bch_pct}%</div>
                </div>
            </div>
        </div>
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 6px; padding: 7px 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div style="font-size: 0.8rem; color: #cbd5e1;">
                💡 <b>Optimal MDP Policy:</b> <span style="color: #38bdf8; font-weight: 600;">{rec_action_label}</span> — <span style="color: #94a3b8;">{snapshot.optimal_policy_rationale}</span>
            </div>
            <div style="font-size: 0.72rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                P_t+h = &pi;_t &middot; P^h
            </div>
        </div>
    </div>
    """
    return html
