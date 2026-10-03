"""Affective Forecasting, Markov Transitions, and MDP Policy Simulation Router."""

from fastapi import APIRouter

from src.core.forecasting_models import (
    AffectiveMacroState,
    InterventionAction,
    HorizonRiskTier,
)
from src.analytics.forecasting import STATE_COORDINATES
from src.api.deps import forecasting_engine
from src.api.v1.schemas import HorizonForecastRequest, PolicySimulationRequest

router = APIRouter(tags=["Affective Forecasting & MDP"])


@router.post("/api/forecast/horizon")
async def project_affective_horizon(req: HorizonForecastRequest):
    """Computes forward Markov transition horizon, hazard indices (EVI/BCH), and optimal MDP policy."""
    snapshot = forecasting_engine.generate_forecast_snapshot(
        valence=req.valence,
        arousal=req.arousal,
        telemetry=req.telemetry,
        horizon_steps=req.horizon_steps or 5,
        step_interval_sec=req.step_interval_sec or 3.0,
    )
    return {"status": "success", "snapshot": snapshot.to_dict()}


@router.post("/api/forecast/simulate-policy")
async def simulate_deescalation_policies(req: PolicySimulationRequest):
    """Simulates counterfactual intervention policies to quantify expected hazard reductions."""
    state = req.current_state or forecasting_engine.classify_macro_state(
        valence=req.valence or 0.0,
        arousal=req.arousal or 0.0,
        cognitive_load=req.telemetry.get("cognitive_load", 0.0) if req.telemetry else 0.0,
        cdri=req.telemetry.get("cdri", 0.0) if req.telemetry else 0.0,
        perclos=req.telemetry.get("perclos", 0.0) if req.telemetry else 0.0,
        pai=req.telemetry.get("pai", 0.0) if req.telemetry else 0.0,
    )
    policies = forecasting_engine.simulate_intervention_policies(
        current_state=state,
        horizon_steps=req.horizon_steps or 5,
        telemetry=req.telemetry,
    )
    return {
        "status": "success",
        "current_state": state,
        "policies": [p.to_dict() for p in policies],
    }


@router.get("/api/forecast/states")
async def get_forecasting_states():
    """Returns discrete affective macro-states and canonical (Valence, Arousal) coordinates."""
    return {
        "status": "online",
        "states": [s.value for s in AffectiveMacroState],
        "state_coordinates": {s: list(coords) for s, coords in STATE_COORDINATES.items()},
        "intervention_actions": [a.value for a in InterventionAction],
        "risk_tiers": [t.value for t in HorizonRiskTier],
    }


@router.get("/api/forecast/config")
async def get_forecasting_config():
    """Retrieves Markov transition baseline parameters, discount factors, and edge SLA specs."""
    return {
        "status": "online",
        "state_order": forecasting_engine.state_order,
        "num_states": forecasting_engine.num_states,
        "gamma_discount": 0.85,
        "step_interval_default_sec": 3.0,
        "sla_target_ms": 2.5,
    }
