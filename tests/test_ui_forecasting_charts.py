"""Unit tests for Phase 14 Plotly forecasting charts and HTML HUD cards."""

import plotly.graph_objects as go

from src.core.forecasting_models import (
    HorizonStepForecast,
    PolicyEvaluationResult,
    HorizonForecastSnapshot,
    HorizonRiskTier,
    InterventionAction,
)
from src.ui.forecasting_charts import (
    render_horizon_trajectory_fan_chart,
    render_markov_transition_heatmap,
    render_hazard_tachometer_gauge,
    render_policy_simulation_comparison_chart,
    render_forecasting_hud_html,
)


def test_render_horizon_trajectory_fan_chart():
    """Verifies generation of multi-step trajectory fan chart with confidence intervals."""
    # 1. Empty trajectory resilience
    empty_fig = render_horizon_trajectory_fan_chart([], height=220)
    assert isinstance(empty_fig, go.Figure)

    # 2. Populated trajectory
    steps = [
        HorizonStepForecast(
            step=1,
            time_offset_sec=3.0,
            predicted_state="Optimal_Flow",
            predicted_valence=0.45,
            predicted_arousal=0.50,
            entropy=1.1,
            escalation_prob=0.08,
            burnout_prob=0.04,
        ),
        HorizonStepForecast(
            step=2,
            time_offset_sec=6.0,
            predicted_state="Optimal_Flow",
            predicted_valence=0.42,
            predicted_arousal=0.48,
            entropy=1.2,
            escalation_prob=0.09,
            burnout_prob=0.05,
        ),
    ]
    fig = render_horizon_trajectory_fan_chart(steps, current_valence=0.5, current_arousal=0.55)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3  # Fan, Valence line, Arousal line


def test_render_markov_transition_heatmap():
    """Verifies generation of 2D Markov transition probability heatmap."""
    # 1. Empty matrix resilience
    empty_fig = render_markov_transition_heatmap({})
    assert isinstance(empty_fig, go.Figure)

    # 2. Populated 3x3 subset matrix
    matrix = {
        "Composure": {"Composure": 0.65, "Cognitive_Strain": 0.20, "Optimal_Flow": 0.15},
        "Cognitive_Strain": {"Composure": 0.15, "Cognitive_Strain": 0.55, "Optimal_Flow": 0.30},
        "Optimal_Flow": {"Composure": 0.25, "Cognitive_Strain": 0.15, "Optimal_Flow": 0.60},
    }
    fig = render_markov_transition_heatmap(matrix, height=280)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_render_hazard_tachometer_gauge():
    """Verifies semicircular hazard gauge rendering across risk tiers."""
    # Stable tier
    fig_stable = render_hazard_tachometer_gauge(0.12, 0.08, HorizonRiskTier.STABLE_EQUILIBRIUM.value)
    assert isinstance(fig_stable, go.Figure)
    assert fig_stable.data[0].type == "indicator"

    # Critical risk tier
    fig_crit = render_hazard_tachometer_gauge(0.85, 0.40, HorizonRiskTier.CRITICAL_COLLAPSE_RISK.value)
    assert isinstance(fig_crit, go.Figure)
    assert fig_crit.data[0].value == 85.0


def test_render_policy_simulation_comparison_chart():
    """Verifies rendering of counterfactual intervention policy comparison bars."""
    # 1. Empty policies resilience
    empty_fig = render_policy_simulation_comparison_chart([])
    assert isinstance(empty_fig, go.Figure)

    # 2. Populated policies
    policies = [
        PolicyEvaluationResult(
            action=InterventionAction.PASSIVE_MONITOR.value,
            action_label="Passive Monitoring",
            expected_valence_recovery=0.0,
            escalation_reduction_percent=0.0,
            burnout_mitigation_percent=0.0,
            stability_score=0.45,
            is_recommended=False,
        ),
        PolicyEvaluationResult(
            action=InterventionAction.ACTIVE_EMPATHY.value,
            action_label="Active Empathic Validation",
            expected_valence_recovery=0.32,
            escalation_reduction_percent=55.0,
            burnout_mitigation_percent=30.0,
            stability_score=0.88,
            is_recommended=True,
        ),
    ]
    fig = render_policy_simulation_comparison_chart(policies)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # Stability, Escalation Red, Burnout Red


def test_render_forecasting_hud_html():
    """Verifies generation of the laboratory-grade forecasting HUD HTML card."""
    snapshot = HorizonForecastSnapshot(
        timestamp=1000.0,
        current_state="Agitated_Frustration",
        current_valence=-0.45,
        current_arousal=0.72,
        horizon_steps=5,
        escalation_velocity_index=0.62,
        burnout_crash_hazard=0.28,
        risk_tier=HorizonRiskTier.HIGH_HAZARD_IMPENDING.value,
        optimal_action="Active_Empathy",
        optimal_policy_rationale="Deploy supportive reflection to defuse escalation.",
    )
    html = render_forecasting_hud_html(snapshot)
    assert isinstance(html, str)
    assert "Agitated Frustration" in html
    assert "62%" in html
    assert "Active Empathy" in html
    assert "HIGH HAZARD IMPENDING" in html
