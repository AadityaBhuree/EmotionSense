"""Remote Photoplethysmography (rPPG) and Autonomic Biometrics Router."""

import time
from fastapi import APIRouter

from src.analytics import BiometricEngine
from src.core.biometric_models import (
    StressClassification,
    PulseMeasurement,
    HRVMetrics,
    RespirationMetrics,
    BiometricTelemetry,
)
from src.api.v1.schemas import BiometricExtractRequest, BiometricStressRequest

router = APIRouter(tags=["Remote Biometrics"])


@router.post("/api/biometrics/rppg")
async def extract_rppg_telemetry(req: BiometricExtractRequest):
    """Extract contact-free optical pulse, HRV, respiration, and autonomic stress from RGB stream."""
    engine = BiometricEngine(fps=req.fps or 30.0)
    for sample in req.rgb_samples:
        if len(sample) >= 3:
            engine.add_rgb_sample(sample[0], sample[1], sample[2])

    bvp = engine.extract_pos_bvp()
    pulse = engine.compute_pulse_from_bvp(bvp)
    rr_intervals = engine.extract_rr_intervals(bvp)
    hrv = engine.compute_hrv_metrics(rr_intervals)
    respiration = engine.estimate_respiration_rate(bvp, rr_intervals)
    stress = engine.compute_autonomic_stress(
        pulse, hrv, respiration, req.valence or 0.0, req.arousal or 0.0, req.vocal_jitter or 0.02
    )
    bvp_history = [float(x) for x in bvp[-120:]] if len(bvp) > 0 else []

    telemetry = BiometricTelemetry(
        timestamp=time.time(),
        pulse=pulse,
        hrv=hrv,
        respiration=respiration,
        autonomic_stress=stress,
        bvp_history=bvp_history,
        rr_intervals_ms=rr_intervals,
        roi_detected=True,
    )
    return {"status": "success", "telemetry": telemetry.to_dict()}


@router.post("/api/biometrics/stress")
async def compute_biometric_stress(req: BiometricStressRequest):
    """Compute autonomic stress classification and sympathetic/parasympathetic tone."""
    pulse = PulseMeasurement(bpm=req.bpm)
    hrv = HRVMetrics(rmssd_ms=req.rmssd_ms, baevsky_stress_index=req.baevsky_si)
    resp = RespirationMetrics(rpm=req.rpm)
    stress = BiometricEngine.compute_autonomic_stress(
        pulse, hrv, resp, req.valence or 0.0, req.arousal or 0.0, req.vocal_jitter or 0.02
    )
    return {"status": "success", "stress": stress.to_dict()}


@router.get("/api/biometrics/status")
async def get_biometric_status():
    """Retrieve biometric engine status, sampling configuration, and supported algorithms."""
    return {
        "status": "online",
        "algorithms": ["Plane-Orthogonal-to-Skin (POS)", "Chrominance (CHROM)", "Respiratory Sinus Arrhythmia (RSA)"],
        "default_fps": 30.0,
        "hrv_metrics": ["SDNN", "RMSSD", "pNN50", "Baevsky_Stress_Index", "HRV_Vitality_Score"],
        "autonomic_classifications": [c.value for c in StressClassification],
    }
