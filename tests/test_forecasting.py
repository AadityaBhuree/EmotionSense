"""Unit tests for Phase 14 Affective Horizon Forecasting & Markov State Transition Engine."""

import pytest
import numpy as np

from src.core.forecasting_models import (
    AffectiveMacroState,
    InterventionAction,
    HorizonRiskTier,
    HorizonForecastSnapshot,
)
from src.analytics.forecasting import (
    AffectiveHorizonForecaster,
    ORDERED_STATES,
)


@pytest.fixture
def forecaster() -> AffectiveHorizonForecaster:
    return AffectiveHorizonForecaster()


def test_canonical_transition_matrix_stochastic(forecaster: AffectiveHorizonForecaster):
    """Verifies that the canonical transition matrix is a valid stochastic matrix."""
    P = forecaster.base_transition_matrix
    assert P.shape == (8, 8)
    assert np.all(P >= 0.0)
    assert np.all(P <= 1.0)
    # Each row must sum to 1.0 within floating point tolerance
    row_sums = P.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(8), rtol=1e-5)


def test_classify_macro_state(forecaster: AffectiveHorizonForecaster):
    """Tests classification of continuous coordinates and telemetry into discrete states."""
    # Composure: low arousal, positive valence
    assert forecaster.classify_macro_state(0.35, 0.15) == AffectiveMacroState.COMPOSURE.value

    # Optimal flow: moderate arousal, positive valence
    assert forecaster.classify_macro_state(0.50, 0.55) == AffectiveMacroState.OPTIMAL_FLOW.value

    # Acute escalation: negative valence, high arousal
    assert forecaster.classify_macro_state(-0.60, 0.80) == AffectiveMacroState.ACUTE_ESCALATION.value

    # Burnout collapse: negative valence, high PERCLOS
    assert forecaster.classify_macro_state(-0.40, 0.10, perclos=0.45) == AffectiveMacroState.BURNOUT_COLLAPSE.value

    # Cognitive strain: high NASA-TLX workload
    assert forecaster.classify_macro_state(-0.05, 0.40, cognitive_load=0.75) == AffectiveMacroState.COGNITIVE_STRAIN.value

    # Deceptive tension: high CDRI
    assert forecaster.classify_macro_state(-0.10, 0.40, cdri=0.78) == AffectiveMacroState.DECEPTIVE_TENSION.value

    # Baseline neutral: default around origin
    assert forecaster.classify_macro_state(0.02, 0.05) == AffectiveMacroState.BASELINE_NEUTRAL.value


def test_modulate_transition_matrix_pulse_surge(forecaster: AffectiveHorizonForecaster):
    """Verifies that physiological pulse surge accelerates escalation transition probabilities."""
    base_P = forecaster.base_transition_matrix
    surge_telemetry = {"pulse_bpm": 105.0, "arousal": 0.70, "pai": 0.65}
    mod_P = forecaster.modulate_transition_matrix(surge_telemetry)

    # Valid stochastic matrix after modulation
    np.testing.assert_allclose(mod_P.sum(axis=1), np.ones(8), rtol=1e-5)

    esc_idx = ORDERED_STATES.index(AffectiveMacroState.ACUTE_ESCALATION.value)
    # From Agitated_Frustration to Acute_Escalation should increase
    frust_idx = ORDERED_STATES.index(AffectiveMacroState.AGITATED_FRUSTRATION.value)
    assert mod_P[frust_idx, esc_idx] > base_P[frust_idx, esc_idx]


def test_project_horizon_steps(forecaster: AffectiveHorizonForecaster):
    """Tests multi-step forward horizon projection."""
    trajectory = forecaster.project_horizon(
        current_state=AffectiveMacroState.AGITATED_FRUSTRATION.value,
        horizon_steps=5,
        step_interval_sec=2.5,
    )

    assert len(trajectory) == 5
    for i, step in enumerate(trajectory):
        assert step.step == i + 1
        assert step.time_offset_sec == round((i + 1) * 2.5, 1)
        assert step.predicted_state in ORDERED_STATES
        assert -1.0 <= step.predicted_valence <= 1.0
        assert 0.0 <= step.predicted_arousal <= 1.0
        assert step.entropy >= 0.0
        # Probabilities sum to 1.0
        total_p = sum(step.state_probabilities.values())
        assert pytest.approx(total_p, 0.01) == 1.0


def test_evaluate_hazards_and_tiers(forecaster: AffectiveHorizonForecaster):
    """Tests calculation of EVI, BCH, and horizon risk tier assignment."""
    # Peaceful trajectory
    calm_trajectory = forecaster.project_horizon(
        current_state=AffectiveMacroState.COMPOSURE.value,
        horizon_steps=4,
    )
    calm_evi, calm_bch, calm_tier = forecaster.evaluate_hazards(calm_trajectory)
    assert calm_evi < 0.25
    assert calm_bch < 0.25
    assert calm_tier == HorizonRiskTier.STABLE_EQUILIBRIUM.value

    # High agitation trajectory with pulse surge
    hostile_trajectory = forecaster.project_horizon(
        current_state=AffectiveMacroState.ACUTE_ESCALATION.value,
        horizon_steps=4,
        telemetry={"pulse_bpm": 115.0, "arousal": 0.85, "pai": 0.75},
    )
    hostile_evi, hostile_bch, hostile_tier = forecaster.evaluate_hazards(hostile_trajectory)
    assert hostile_evi > calm_evi
    assert hostile_tier in [HorizonRiskTier.HIGH_HAZARD_IMPENDING.value, HorizonRiskTier.CRITICAL_COLLAPSE_RISK.value]


def test_simulate_intervention_policies(forecaster: AffectiveHorizonForecaster):
    """Tests MDP counterfactual policy evaluations and optimal recommendation selection."""
    policies = forecaster.simulate_intervention_policies(
        current_state=AffectiveMacroState.ACUTE_ESCALATION.value,
        horizon_steps=5,
        telemetry={"pulse_bpm": 105.0, "arousal": 0.80},
    )

    assert len(policies) == 5
    actions = [p.action for p in policies]
    assert InterventionAction.PASSIVE_MONITOR.value in actions
    assert InterventionAction.ACTIVE_EMPATHY.value in actions
    assert InterventionAction.TEMPO_DECELERATION.value in actions
    assert InterventionAction.COGNITIVE_OFFLOADING.value in actions
    assert InterventionAction.PHYSIOLOGICAL_RESET.value in actions

    # Active empathy should achieve significant escalation reduction on acute escalation
    empathy_policy = next(p for p in policies if p.action == InterventionAction.ACTIVE_EMPATHY.value)
    assert empathy_policy.escalation_reduction_percent > 10.0

    # Exactly one policy should be marked as recommended
    recommended_count = sum(1 for p in policies if p.is_recommended)
    assert recommended_count == 1


def test_generate_forecast_snapshot_serialization(forecaster: AffectiveHorizonForecaster):
    """Verifies complete forecast snapshot creation and dictionary serialization."""
    snapshot = forecaster.generate_forecast_snapshot(
        valence=-0.50,
        arousal=0.75,
        telemetry={"pulse_bpm": 92.0, "cognitive_load": 0.60, "pai": 0.40},
        horizon_steps=5,
        step_interval_sec=3.0,
    )

    assert isinstance(snapshot, HorizonForecastSnapshot)
    assert snapshot.current_state in [AffectiveMacroState.ACUTE_ESCALATION.value, AffectiveMacroState.AGITATED_FRUSTRATION.value]
    assert len(snapshot.forecast_trajectory) == 5
    assert len(snapshot.policy_options) == 5
    assert snapshot.optimal_action != ""

    snap_dict = snapshot.to_dict()
    assert "forecast_trajectory" in snap_dict
    assert "transition_matrix" in snap_dict
    assert "policy_options" in snap_dict
    assert snap_dict["current_valence"] == -0.50
