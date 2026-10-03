"""Oculomotor and Cognitive Workload Router."""

import numpy as np
from fastapi import APIRouter

from src.analytics import OculomotorEngine
from src.core.cognitive_models import (
    WorkloadTier,
    PupillometryMetrics,
    BlinkDynamics,
    GazeTelemetry,
)
from src.api.deps import oculomotor_engine
from src.api.v1.schemas import OculometricsRequest, CognitiveWorkloadRequest

router = APIRouter(tags=["Cognitive Workload"])


@router.post("/api/cognitive/oculometrics")
async def evaluate_oculometrics(req: OculometricsRequest):
    """Evaluate eye landmarks, pupil dilation ratio, blink rate, and gaze dynamics."""
    l_pts = np.array(req.left_eye_points, dtype=np.float64)
    r_pts = np.array(req.right_eye_points, dtype=np.float64)
    snapshot = oculomotor_engine.process_frame(
        left_eye_pts=l_pts,
        right_eye_pts=r_pts,
        pir_sample=req.pir_sample or 0.42,
        yaw_deg=req.yaw_deg or 0.0,
        pitch_deg=req.pitch_deg or 0.0,
        autonomic_strain=req.autonomic_strain or 0.30,
    )
    return {"status": "success", "snapshot": snapshot.to_dict()}


@router.post("/api/cognitive/workload")
async def compute_cognitive_workload_endpoint(req: CognitiveWorkloadRequest):
    """Compute unified Cognitive Workload Index and NASA-TLX dimensional estimates."""
    pupil = PupillometryMetrics(cognitive_pupillary_response=req.cpr)
    blinks = BlinkDynamics(perclos=req.perclos, blink_suppressed=req.blink_suppressed or False)
    gaze = GazeTelemetry(
        fixation_duration_ms=req.fixation_duration_ms or 350.0,
        dispersion_area=req.dispersion_area or 0.12,
    )
    workload = OculomotorEngine.compute_cognitive_workload(
        pupil,
        blinks,
        gaze,
        autonomic_strain=req.autonomic_strain or 0.30,
        speech_pause_ratio=req.speech_pause_ratio or 0.20,
    )
    return {"status": "success", "workload": workload.to_dict()}


@router.get("/api/cognitive/config")
async def get_cognitive_config():
    """Retrieve cognitive engine calibration thresholds, EAR baseline, and NASA-TLX specifications."""
    return {
        "status": "online",
        "ear_closed_threshold": oculomotor_engine.ear_closed_threshold,
        "baseline_pir": oculomotor_engine.baseline_pir,
        "saccade_velocity_threshold_deg_s": oculomotor_engine.saccade_velocity_threshold,
        "workload_tiers": [t.value for t in WorkloadTier],
        "nasa_tlx_dimensions": ["mental_demand", "temporal_demand", "effort", "frustration"],
    }
