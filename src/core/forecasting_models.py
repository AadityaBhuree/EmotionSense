"""Phase 14: Affective Horizon Forecasting & Markov State Transition Dynamics Data Models.

Defines domain contracts for discrete affective macro-states, multi-step horizon
probability projections, transition matrix structures, escalation & burnout hazard indices,
and Markov Decision Process (MDP) counterfactual policy evaluations.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any


class AffectiveMacroState(str, Enum):
    """Discrete affective macro-states for Markov transition modeling."""
    COMPOSURE = "Composure"                     # Low arousal, positive valence
    OPTIMAL_FLOW = "Optimal_Flow"               # Moderate arousal, positive valence, high engagement
    COGNITIVE_STRAIN = "Cognitive_Strain"       # High workload, neutral/negative valence
    AGITATED_FRUSTRATION = "Agitated_Frustration" # High arousal, negative valence, high restlessness
    ACUTE_ESCALATION = "Acute_Escalation"       # Surging anger/hostility, high cardiac pulse surge
    BURNOUT_COLLAPSE = "Burnout_Collapse"       # Plunging valence, low arousal, high fatigue
    DECEPTIVE_TENSION = "Deceptive_Tension"     # Elevated CDRI, micro-leakage & voice stress
    BASELINE_NEUTRAL = "Baseline_Neutral"       # Equilibrium reference point


class InterventionAction(str, Enum):
    """Discrete intervention actions available in the Markov Decision Process (MDP)."""
    PASSIVE_MONITOR = "Passive_Monitor"         # Baseline observation without interference
    ACTIVE_EMPATHY = "Active_Empathy"           # Supportive active listening and emotional validation
    TEMPO_DECELERATION = "Tempo_Deceleration"   # Slowing conversation pace, acoustic de-escalation
    COGNITIVE_OFFLOADING = "Cognitive_Offloading" # Reducing task complexity, lowering demands
    PHYSIOLOGICAL_RESET = "Physiological_Reset" # Prompting breathing reset or micro-pause


class HorizonRiskTier(str, Enum):
    """Overall hazard classification for the projected emotional horizon."""
    STABLE_EQUILIBRIUM = "Stable_Equilibrium"       # EVI < 0.25 and BCH < 0.25
    ELEVATED_DRIFT = "Elevated_Drift"               # EVI 0.25 - 0.50 or BCH 0.25 - 0.50
    HIGH_HAZARD_IMPENDING = "High_Hazard_Impending" # EVI 0.50 - 0.75 or BCH 0.50 - 0.75
    CRITICAL_COLLAPSE_RISK = "Critical_Collapse_Risk" # EVI >= 0.75 or BCH >= 0.75


@dataclass
class HorizonStepForecast:
    """Predicted affective probability distribution and expected coordinates at step h."""
    step: int = 1
    time_offset_sec: float = 3.0
    state_probabilities: Dict[str, float] = field(default_factory=dict)
    predicted_state: str = AffectiveMacroState.BASELINE_NEUTRAL.value
    predicted_valence: float = 0.0
    predicted_arousal: float = 0.0
    entropy: float = 1.2
    escalation_prob: float = 0.05
    burnout_prob: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyEvaluationResult:
    """Evaluation metrics for a simulated counterfactual intervention action."""
    action: str = InterventionAction.PASSIVE_MONITOR.value
    action_label: str = "Passive Monitoring"
    expected_valence_recovery: float = 0.0       # Expected delta in valence over baseline
    escalation_reduction_percent: float = 0.0    # Percentage reduction in escalation probability
    burnout_mitigation_percent: float = 0.0      # Percentage reduction in burnout probability
    stability_score: float = 0.75                # Expected horizon equilibrium score (0.0 to 1.0)
    is_recommended: bool = False
    clinical_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HorizonForecastSnapshot:
    """Comprehensive snapshot of affective forecasting, hazard indices, and optimal policies."""
    timestamp: float = 0.0
    current_state: str = AffectiveMacroState.BASELINE_NEUTRAL.value
    current_valence: float = 0.0
    current_arousal: float = 0.0
    horizon_steps: int = 5
    forecast_trajectory: List[HorizonStepForecast] = field(default_factory=list)
    escalation_velocity_index: float = 0.08      # EVI (0.0 to 1.0)
    burnout_crash_hazard: float = 0.05           # BCH (0.0 to 1.0)
    risk_tier: str = HorizonRiskTier.STABLE_EQUILIBRIUM.value
    transition_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    policy_options: List[PolicyEvaluationResult] = field(default_factory=list)
    optimal_action: str = InterventionAction.PASSIVE_MONITOR.value
    optimal_policy_rationale: str = "Affective trajectory is stable in equilibrium."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
