"""Forensic Veracity, Micro-Leakage, and Credibility Router."""

from fastapi import APIRouter

from src.core.credibility_models import (
    VeracityTier,
    DeceptionFlag,
    VoiceStressProfile,
)
from src.api.deps import credibility_engine
from src.api.v1.schemas import ForensicCredibilityRequest, MicroLeakageAnalysisRequest

router = APIRouter(tags=["Forensic Veracity & Credibility"])


@router.post("/api/forensic/credibility")
async def evaluate_forensic_credibility(req: ForensicCredibilityRequest):
    """Fuses multi-channel affective markers into the Credibility & Deception Risk Index (CDRI)."""
    fac_metrics, leakages, fac_flags = credibility_engine.compute_facial_veracity(
        action_units=req.action_units,
        macro_emotion=req.macro_emotion or "Neutral",
        valence=req.valence or 0.0,
    )
    vocal = VoiceStressProfile(
        stress_index=req.voice_stress_index or 0.18,
        response_latency_sec=req.response_latency_sec or 0.45,
        is_voice_stressed=(req.voice_stress_index or 0.18) >= credibility_engine.voice_stress_threshold,
    )
    vocal_flags = [DeceptionFlag.ACOUSTIC_MICRO_TREMOR.value] if vocal.is_voice_stressed else []
    if (req.response_latency_sec or 0.45) > 2.5:
        vocal_flags.append(DeceptionFlag.LATENCY_ELONGATION.value)

    snapshot = credibility_engine.fuse_credibility_assessment(
        facial_metrics=fac_metrics,
        voice_stress=vocal,
        pupil_cpr=req.pupil_cpr or 1.0,
        pacifying_adaptor_active=req.pacifying_adaptor_active or False,
        pulse_bpm=req.pulse_bpm or 72.0,
        detected_flags=fac_flags + vocal_flags,
    )
    return {"status": "success", "snapshot": snapshot.to_dict()}


@router.post("/api/forensic/leakage")
async def evaluate_micro_leakage(req: MicroLeakageAnalysisRequest):
    """Evaluates facial Action Units for Duchenne smile incongruence and transient micro-expression flashes."""
    fac_metrics, leakages, flags = credibility_engine.compute_facial_veracity(
        action_units=req.action_units,
        macro_emotion=req.macro_emotion or "Neutral",
        valence=req.valence or 0.0,
    )
    return {
        "status": "success",
        "facial_veracity": fac_metrics.to_dict(),
        "leakages": [lk.to_dict() for lk in leakages],
        "flags": flags,
    }


@router.get("/api/forensic/config")
async def get_forensic_config():
    """Retrieves forensic veracity tiers, deception flags, and calibration thresholds."""
    return {
        "status": "online",
        "leakage_flash_max_ms": credibility_engine.leakage_flash_max_ms,
        "duchenne_threshold": credibility_engine.duchenne_threshold,
        "voice_stress_threshold": credibility_engine.voice_stress_threshold,
        "veracity_tiers": [t.value for t in VeracityTier],
        "deception_flags": [f.value for f in DeceptionFlag],
        "weights": {
            "facial": credibility_engine.WEIGHT_FACIAL,
            "vocal": credibility_engine.WEIGHT_VOCAL,
            "pupillometric": credibility_engine.WEIGHT_PUPILLOMETRIC,
            "somatosensory": credibility_engine.WEIGHT_SOMATOSENSORY,
            "autonomic": credibility_engine.WEIGHT_AUTONOMIC,
        },
    }
