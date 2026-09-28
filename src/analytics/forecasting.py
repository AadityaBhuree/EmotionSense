"""Phase 14: Affective Horizon Forecasting & Markov State Transition Engine.

Implements discrete-state Markov transition dynamics, multi-step horizon probability
distribution forecasting, continuous telemetry covariate modulation, and Markov
Decision Process (MDP) counterfactual intervention policy simulation.
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from src.core.forecasting_models import (
    AffectiveMacroState,
    InterventionAction,
    HorizonRiskTier,
    HorizonStepForecast,
    PolicyEvaluationResult,
    HorizonForecastSnapshot,
)

# Canonical coordinates for discrete macro-states (Valence, Arousal)
STATE_COORDINATES: Dict[str, Tuple[float, float]] = {
    AffectiveMacroState.COMPOSURE.value: (0.45, 0.20),
    AffectiveMacroState.OPTIMAL_FLOW.value: (0.65, 0.55),
    AffectiveMacroState.COGNITIVE_STRAIN.value: (-0.15, 0.40),
    AffectiveMacroState.AGITATED_FRUSTRATION.value: (-0.55, 0.70),
    AffectiveMacroState.ACUTE_ESCALATION.value: (-0.75, 0.90),
    AffectiveMacroState.BURNOUT_COLLAPSE.value: (-0.70, 0.15),
    AffectiveMacroState.DECEPTIVE_TENSION.value: (-0.30, 0.60),
    AffectiveMacroState.BASELINE_NEUTRAL.value: (0.00, 0.00),
}

ORDERED_STATES: List[str] = [
    AffectiveMacroState.COMPOSURE.value,
    AffectiveMacroState.OPTIMAL_FLOW.value,
    AffectiveMacroState.COGNITIVE_STRAIN.value,
    AffectiveMacroState.AGITATED_FRUSTRATION.value,
    AffectiveMacroState.ACUTE_ESCALATION.value,
    AffectiveMacroState.BURNOUT_COLLAPSE.value,
    AffectiveMacroState.DECEPTIVE_TENSION.value,
    AffectiveMacroState.BASELINE_NEUTRAL.value,
]

STATE_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(ORDERED_STATES)}


class AffectiveHorizonForecaster:
    """Predictive Markov transition dynamics and MDP intervention policy sentinel."""

    def __init__(self):
        self.state_order = ORDERED_STATES
        self.num_states = len(ORDERED_STATES)
        self.base_transition_matrix = self._build_canonical_transition_matrix()

    def _build_canonical_transition_matrix(self) -> np.ndarray:
        """Constructs canonical psychological transition probability matrix P in R^{8x8}."""
        # Initial priors reflecting inertia and natural emotional migration
        # [Composure, Optimal_Flow, Cognitive_Strain, Agitated_Frustration, Acute_Escalation, Burnout_Collapse, Deceptive_Tension, Baseline_Neutral]
        P = np.array([
            [0.55, 0.25, 0.05, 0.02, 0.01, 0.01, 0.01, 0.10],  # From Composure
            [0.20, 0.50, 0.15, 0.05, 0.01, 0.02, 0.01, 0.06],  # From Optimal_Flow
            [0.05, 0.10, 0.40, 0.20, 0.05, 0.12, 0.03, 0.05],  # From Cognitive_Strain
            [0.02, 0.02, 0.10, 0.45, 0.25, 0.08, 0.03, 0.05],  # From Agitated_Frustration
            [0.01, 0.01, 0.05, 0.25, 0.55, 0.05, 0.02, 0.06],  # From Acute_Escalation
            [0.08, 0.02, 0.15, 0.05, 0.02, 0.50, 0.02, 0.16],  # From Burnout_Collapse
            [0.05, 0.05, 0.15, 0.20, 0.10, 0.05, 0.35, 0.05],  # From Deceptive_Tension
            [0.25, 0.20, 0.12, 0.08, 0.02, 0.03, 0.02, 0.28],  # From Baseline_Neutral
        ], dtype=np.float64)

        # Normalize rows to ensure valid stochastic matrix
        return P / P.sum(axis=1, keepdims=True)

    def classify_macro_state(
        self,
        valence: float,
        arousal: float,
        cognitive_load: float = 0.0,
        cdri: float = 0.0,
        perclos: float = 0.0,
        pai: float = 0.0,
    ) -> str:
        """Classifies continuous telemetry signals into the most descriptive discrete macro-state."""
        # Deception condition takes precedence if veracity risk is high
        if cdri >= 0.65:
            return AffectiveMacroState.DECEPTIVE_TENSION.value

        # High cognitive overload with low/moderate fatigue
        if cognitive_load >= 0.68 and perclos < 0.35:
            return AffectiveMacroState.COGNITIVE_STRAIN.value

        # Severe burnout or exhaustion crash
        if perclos >= 0.35 and valence <= -0.20:
            return AffectiveMacroState.BURNOUT_COLLAPSE.value

        # Acute hostility and aggressive escalation
        if valence <= -0.45 and arousal >= 0.65:
            return AffectiveMacroState.ACUTE_ESCALATION.value

        # Restless frustration and agitation
        if (valence <= -0.25 and arousal >= 0.45) or (pai >= 0.60 and valence < 0.0):
            return AffectiveMacroState.AGITATED_FRUSTRATION.value

        # Optimal engagement and flow state
        if valence >= 0.20 and 0.35 <= arousal <= 0.80:
            return AffectiveMacroState.OPTIMAL_FLOW.value

        # Composure and relaxed satisfaction
        if valence >= 0.15 and arousal < 0.35:
            return AffectiveMacroState.COMPOSURE.value

        # Default fallback to neutral equilibrium
        return AffectiveMacroState.BASELINE_NEUTRAL.value

    def modulate_transition_matrix(
        self,
        telemetry: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Applies continuous physiological and acoustic covariates to modulate transition dynamics."""
        P = self.base_transition_matrix.copy()
        if not telemetry:
            return P

        pulse_bpm = telemetry.get("pulse_bpm", 72.0)
        cognitive_load = telemetry.get("cognitive_load", 0.3)
        perclos = telemetry.get("perclos", 0.1)
        cdri = telemetry.get("cdri", 0.1)
        pai = telemetry.get("pai", 0.1)
        arousal = telemetry.get("arousal", 0.0)

        # Pulse surge and high agitation accelerate Acute Escalation probability
        if pulse_bpm > 85.0 or pai > 0.50 or arousal > 0.50:
            surge_mult = 1.0 + min(2.0, ((pulse_bpm - 72.0) / 30.0) + pai)
            esc_idx = STATE_TO_IDX[AffectiveMacroState.ACUTE_ESCALATION.value]
            frust_idx = STATE_TO_IDX[AffectiveMacroState.AGITATED_FRUSTRATION.value]
            P[:, esc_idx] *= surge_mult
            P[:, frust_idx] *= (surge_mult * 0.8)

        # Prolonged cognitive load shifts transitions toward Burnout and Strain
        if cognitive_load > 0.60:
            strain_idx = STATE_TO_IDX[AffectiveMacroState.COGNITIVE_STRAIN.value]
            P[:, strain_idx] *= 1.45

        # Drowsiness/PERCLOS elevates Burnout Collapse probability
        if perclos > 0.25:
            burnout_idx = STATE_TO_IDX[AffectiveMacroState.BURNOUT_COLLAPSE.value]
            P[:, burnout_idx] *= (1.0 + perclos * 2.5)

        # CDRI elevation diverts transitions toward Deceptive Tension
        if cdri > 0.45:
            dec_idx = STATE_TO_IDX[AffectiveMacroState.DECEPTIVE_TENSION.value]
            P[:, dec_idx] *= (1.0 + cdri * 2.0)

        # Re-normalize rows to maintain stochastic integrity
        return P / P.sum(axis=1, keepdims=True)

    def project_horizon(
        self,
        current_state: str,
        horizon_steps: int = 5,
        step_interval_sec: float = 3.0,
        telemetry: Optional[Dict[str, float]] = None,
        custom_P: Optional[np.ndarray] = None,
    ) -> List[HorizonStepForecast]:
        """Projects multi-step forward probability distributions and trajectory coordinates."""
        P = custom_P if custom_P is not None else self.modulate_transition_matrix(telemetry)

        # Initial one-hot state vector
        pi = np.zeros(self.num_states, dtype=np.float64)
        init_idx = STATE_TO_IDX.get(current_state, STATE_TO_IDX[AffectiveMacroState.BASELINE_NEUTRAL.value])
        pi[init_idx] = 1.0

        forecasts: List[HorizonStepForecast] = []
        current_pi = pi.copy()

        for step in range(1, horizon_steps + 1):
            current_pi = np.dot(current_pi, P)
            # Ensure numerical stability
            current_pi = np.clip(current_pi, 0.0, 1.0)
            current_pi = current_pi / current_pi.sum()

            state_probs = {self.state_order[i]: float(current_pi[i]) for i in range(self.num_states)}
            pred_state = self.state_order[int(np.argmax(current_pi))]

            # Compute expected Valence and Arousal
            exp_valence = sum(current_pi[i] * STATE_COORDINATES[self.state_order[i]][0] for i in range(self.num_states))
            exp_arousal = sum(current_pi[i] * STATE_COORDINATES[self.state_order[i]][1] for i in range(self.num_states))

            # Shannon entropy of prediction certainty
            entropy = -sum(p * math.log2(p + 1e-9) for p in current_pi)

            # Hazard probabilities
            esc_prob = state_probs.get(AffectiveMacroState.ACUTE_ESCALATION.value, 0.0) + (
                0.5 * state_probs.get(AffectiveMacroState.AGITATED_FRUSTRATION.value, 0.0)
            )
            burnout_prob = state_probs.get(AffectiveMacroState.BURNOUT_COLLAPSE.value, 0.0) + (
                0.4 * state_probs.get(AffectiveMacroState.COGNITIVE_STRAIN.value, 0.0)
            )

            forecasts.append(HorizonStepForecast(
                step=step,
                time_offset_sec=round(step * step_interval_sec, 1),
                state_probabilities=state_probs,
                predicted_state=pred_state,
                predicted_valence=round(float(exp_valence), 3),
                predicted_arousal=round(float(exp_arousal), 3),
                entropy=round(float(entropy), 3),
                escalation_prob=round(float(esc_prob), 3),
                burnout_prob=round(float(burnout_prob), 3),
            ))

        return forecasts

    def evaluate_hazards(
        self,
        forecast_trajectory: List[HorizonStepForecast],
        gamma: float = 0.85,
    ) -> Tuple[float, float, str]:
        """Calculates Escalation Velocity Index (EVI), Burnout Crash Hazard (BCH), and tier."""
        if not forecast_trajectory:
            return 0.0, 0.0, HorizonRiskTier.STABLE_EQUILIBRIUM.value

        evi = 0.0
        bch = 0.0
        normalizer = sum(gamma ** i for i in range(len(forecast_trajectory)))

        for i, step in enumerate(forecast_trajectory):
            weight = gamma ** i
            evi += weight * step.escalation_prob
            bch += weight * step.burnout_prob

        evi = min(1.0, max(0.0, evi / normalizer))
        bch = min(1.0, max(0.0, bch / normalizer))

        max_hazard = max(evi, bch)
        if max_hazard < 0.25:
            tier = HorizonRiskTier.STABLE_EQUILIBRIUM.value
        elif max_hazard < 0.50:
            tier = HorizonRiskTier.ELEVATED_DRIFT.value
        elif max_hazard < 0.75:
            tier = HorizonRiskTier.HIGH_HAZARD_IMPENDING.value
        else:
            tier = HorizonRiskTier.CRITICAL_COLLAPSE_RISK.value

        return round(evi, 3), round(bch, 3), tier

    def simulate_intervention_policies(
        self,
        current_state: str,
        horizon_steps: int = 5,
        telemetry: Optional[Dict[str, float]] = None,
    ) -> List[PolicyEvaluationResult]:
        """Simulates MDP counterfactual policy outcomes across 5 distinct intervention actions."""
        base_P = self.modulate_transition_matrix(telemetry)
        base_forecast = self.project_horizon(current_state, horizon_steps, custom_P=base_P)
        base_evi, base_bch, _ = self.evaluate_hazards(base_forecast)
        base_val = base_forecast[-1].predicted_valence if base_forecast else 0.0

        policies: List[PolicyEvaluationResult] = []

        # 1. PASSIVE_MONITOR (Baseline)
        policies.append(PolicyEvaluationResult(
            action=InterventionAction.PASSIVE_MONITOR.value,
            action_label="Passive Monitoring",
            expected_valence_recovery=0.0,
            escalation_reduction_percent=0.0,
            burnout_mitigation_percent=0.0,
            stability_score=round(max(0.1, 1.0 - max(base_evi, base_bch)), 2),
            is_recommended=False,
            clinical_rationale="Continuous passive tracking without external conversational adjustment.",
        ))

        # 2. ACTIVE_EMPATHY
        # Dampens hostile transitions, re-routes toward Composure and Neutral
        empathy_P = base_P.copy()
        esc_idx = STATE_TO_IDX[AffectiveMacroState.ACUTE_ESCALATION.value]
        frust_idx = STATE_TO_IDX[AffectiveMacroState.AGITATED_FRUSTRATION.value]
        comp_idx = STATE_TO_IDX[AffectiveMacroState.COMPOSURE.value]
        neut_idx = STATE_TO_IDX[AffectiveMacroState.BASELINE_NEUTRAL.value]

        empathy_P[:, esc_idx] *= 0.35
        empathy_P[:, frust_idx] *= 0.45
        empathy_P[:, comp_idx] += 0.25
        empathy_P[:, neut_idx] += 0.20
        empathy_P = empathy_P / empathy_P.sum(axis=1, keepdims=True)

        empathy_forecast = self.project_horizon(current_state, horizon_steps, custom_P=empathy_P)
        emp_evi, emp_bch, _ = self.evaluate_hazards(empathy_forecast)
        emp_val = empathy_forecast[-1].predicted_valence if empathy_forecast else 0.0

        esc_red = max(0.0, (base_evi - emp_evi) / (base_evi + 1e-6)) * 100.0
        policies.append(PolicyEvaluationResult(
            action=InterventionAction.ACTIVE_EMPATHY.value,
            action_label="Active Empathic Validation",
            expected_valence_recovery=round(max(0.0, emp_val - base_val), 3),
            escalation_reduction_percent=round(esc_red, 1),
            burnout_mitigation_percent=round(max(0.0, (base_bch - emp_bch) / (base_bch + 1e-6)) * 100.0, 1),
            stability_score=round(max(0.1, 1.0 - max(emp_evi, emp_bch)), 2),
            is_recommended=False,
            clinical_rationale="Deploy supportive reflection and emotional validation to defuse escalating tension.",
        ))

        # 3. TEMPO_DECELERATION
        # Drops high arousal, lowers agitation
        tempo_P = base_P.copy()
        tempo_P[:, esc_idx] *= 0.50
        tempo_P[:, frust_idx] *= 0.60
        tempo_P[:, comp_idx] += 0.30
        tempo_P = tempo_P / tempo_P.sum(axis=1, keepdims=True)

        tempo_forecast = self.project_horizon(current_state, horizon_steps, custom_P=tempo_P)
        tempo_evi, tempo_bch, _ = self.evaluate_hazards(tempo_forecast)
        tempo_val = tempo_forecast[-1].predicted_valence if tempo_forecast else 0.0

        policies.append(PolicyEvaluationResult(
            action=InterventionAction.TEMPO_DECELERATION.value,
            action_label="Conversational Tempo Deceleration",
            expected_valence_recovery=round(max(0.0, tempo_val - base_val), 3),
            escalation_reduction_percent=round(max(0.0, (base_evi - tempo_evi) / (base_evi + 1e-6)) * 100.0, 1),
            burnout_mitigation_percent=round(max(0.0, (base_bch - tempo_bch) / (base_bch + 1e-6)) * 100.0, 1),
            stability_score=round(max(0.1, 1.0 - max(tempo_evi, tempo_bch)), 2),
            is_recommended=False,
            clinical_rationale="Introduce measured acoustic pauses and slow speaking cadence to reduce autonomic drive.",
        ))

        # 4. COGNITIVE_OFFLOADING
        # Drastically reduces Cognitive Strain and Burnout Collapse
        offload_P = base_P.copy()
        strain_idx = STATE_TO_IDX[AffectiveMacroState.COGNITIVE_STRAIN.value]
        burn_idx = STATE_TO_IDX[AffectiveMacroState.BURNOUT_COLLAPSE.value]
        flow_idx = STATE_TO_IDX[AffectiveMacroState.OPTIMAL_FLOW.value]

        offload_P[:, strain_idx] *= 0.25
        offload_P[:, burn_idx] *= 0.35
        offload_P[:, flow_idx] += 0.30
        offload_P = offload_P / offload_P.sum(axis=1, keepdims=True)

        offload_forecast = self.project_horizon(current_state, horizon_steps, custom_P=offload_P)
        offload_evi, offload_bch, _ = self.evaluate_hazards(offload_forecast)
        offload_val = offload_forecast[-1].predicted_valence if offload_forecast else 0.0

        bch_red = max(0.0, (base_bch - offload_bch) / (base_bch + 1e-6)) * 100.0
        policies.append(PolicyEvaluationResult(
            action=InterventionAction.COGNITIVE_OFFLOADING.value,
            action_label="Cognitive Demand Offloading",
            expected_valence_recovery=round(max(0.0, offload_val - base_val), 3),
            escalation_reduction_percent=round(max(0.0, (base_evi - offload_evi) / (base_evi + 1e-6)) * 100.0, 1),
            burnout_mitigation_percent=round(bch_red, 1),
            stability_score=round(max(0.1, 1.0 - max(offload_evi, offload_bch)), 2),
            is_recommended=False,
            clinical_rationale="Simplify conversational complexity and delegate cognitive subtasks to prevent burnout.",
        ))

        # 5. PHYSIOLOGICAL_RESET
        # High impact reset for both escalation and exhaustion
        reset_P = base_P.copy()
        reset_P[:, esc_idx] *= 0.30
        reset_P[:, burn_idx] *= 0.40
        reset_P[:, neut_idx] += 0.40
        reset_P = reset_P / reset_P.sum(axis=1, keepdims=True)

        reset_forecast = self.project_horizon(current_state, horizon_steps, custom_P=reset_P)
        res_evi, res_bch, _ = self.evaluate_hazards(reset_forecast)
        res_val = reset_forecast[-1].predicted_valence if reset_forecast else 0.0

        policies.append(PolicyEvaluationResult(
            action=InterventionAction.PHYSIOLOGICAL_RESET.value,
            action_label="Autonomic Physiological Reset",
            expected_valence_recovery=round(max(0.0, res_val - base_val), 3),
            escalation_reduction_percent=round(max(0.0, (base_evi - res_evi) / (base_evi + 1e-6)) * 100.0, 1),
            burnout_mitigation_percent=round(max(0.0, (base_bch - res_bch) / (base_bch + 1e-6)) * 100.0, 1),
            stability_score=round(max(0.1, 1.0 - max(res_evi, res_bch)), 2),
            is_recommended=False,
            clinical_rationale="Initiate a 15-second resonant breathing micro-pause or postural alignment reset.",
        ))

        # Select mathematically optimal policy maximizing stability and hazard reduction
        best_policy = max(policies, key=lambda p: (p.stability_score * 0.5 + (p.expected_valence_recovery * 0.5)))
        best_policy.is_recommended = True

        return policies

    def generate_forecast_snapshot(
        self,
        valence: float,
        arousal: float,
        telemetry: Optional[Dict[str, float]] = None,
        horizon_steps: int = 5,
        step_interval_sec: float = 3.0,
    ) -> HorizonForecastSnapshot:
        """Synthesizes comprehensive HorizonForecastSnapshot covering predictions and MDP policies."""
        macro_state = self.classify_macro_state(
            valence=valence,
            arousal=arousal,
            cognitive_load=telemetry.get("cognitive_load", 0.0) if telemetry else 0.0,
            cdri=telemetry.get("cdri", 0.0) if telemetry else 0.0,
            perclos=telemetry.get("perclos", 0.0) if telemetry else 0.0,
            pai=telemetry.get("pai", 0.0) if telemetry else 0.0,
        )

        P = self.modulate_transition_matrix(telemetry)
        forecast_trajectory = self.project_horizon(
            current_state=macro_state,
            horizon_steps=horizon_steps,
            step_interval_sec=step_interval_sec,
            custom_P=P,
        )

        evi, bch, risk_tier = self.evaluate_hazards(forecast_trajectory)
        policy_options = self.simulate_intervention_policies(
            current_state=macro_state,
            horizon_steps=horizon_steps,
            telemetry=telemetry,
        )

        recommended_policy = next((p for p in policy_options if p.is_recommended), policy_options[0])

        # Serialize transition matrix as nested dictionary for inspection
        trans_dict: Dict[str, Dict[str, float]] = {}
        for i, src in enumerate(self.state_order):
            trans_dict[src] = {dst: round(float(P[i, j]), 3) for j, dst in enumerate(self.state_order)}

        return HorizonForecastSnapshot(
            timestamp=float(telemetry.get("timestamp", 0.0)) if telemetry else 0.0,
            current_state=macro_state,
            current_valence=round(valence, 3),
            current_arousal=round(arousal, 3),
            horizon_steps=horizon_steps,
            forecast_trajectory=forecast_trajectory,
            escalation_velocity_index=evi,
            burnout_crash_hazard=bch,
            risk_tier=risk_tier,
            transition_matrix=trans_dict,
            policy_options=policy_options,
            optimal_action=recommended_policy.action,
            optimal_policy_rationale=recommended_policy.clinical_rationale,
        )
