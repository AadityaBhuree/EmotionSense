"""Somatosensory Kinematics and Postural Ergonomics Router."""

from fastapi import APIRouter

from src.analytics import SomatosensoryEngine
from src.core.somatosensory_models import (
    PostureState,
    AdaptorType,
    AdaptorCategory,
    PsychomotorTier,
    PosturalMetrics,
    MicroGestureAdaptor,
    FidgetingDynamics,
)
from src.api.deps import somatosensory_engine
from src.api.v1.schemas import SomatosensoryKinematicsRequest, PsychomotorAgitationRequest

router = APIRouter(tags=["Somatosensory Kinematics"])


@router.post("/api/somatosensory/kinematics")
async def evaluate_somatosensory_kinematics(req: SomatosensoryKinematicsRequest):
    """Evaluate upper-body posture, micro-gesture self-touch adaptors, and kinetic fidgeting."""
    ls = (req.left_shoulder[0], req.left_shoulder[1]) if len(req.left_shoulder) >= 2 else (0.30, 0.60)
    rs = (req.right_shoulder[0], req.right_shoulder[1]) if len(req.right_shoulder) >= 2 else (0.70, 0.60)
    nose = (req.nose[0], req.nose[1]) if len(req.nose) >= 2 else (0.50, 0.30)
    le = (req.left_ear[0], req.left_ear[1]) if req.left_ear and len(req.left_ear) >= 2 else None
    re = (req.right_ear[0], req.right_ear[1]) if req.right_ear and len(req.right_ear) >= 2 else None
    lh = (req.left_hand[0], req.left_hand[1]) if req.left_hand and len(req.left_hand) >= 2 else None
    rh = (req.right_hand[0], req.right_hand[1]) if req.right_hand and len(req.right_hand) >= 2 else None

    snapshot = somatosensory_engine.process_frame(
        left_shoulder=ls,
        right_shoulder=rs,
        nose=nose,
        left_ear=le,
        right_ear=re,
        left_hand=lh,
        right_hand=rh,
        autonomic_stress=req.autonomic_stress or 0.25,
        cognitive_workload=req.cognitive_workload or 0.30,
        acoustic_jitter=req.acoustic_jitter or 0.02,
        speech_active=req.speech_active if req.speech_active is not None else True,
    )
    return {"status": "success", "snapshot": snapshot.to_dict()}


@router.post("/api/somatosensory/agitation")
async def compute_psychomotor_agitation_endpoint(req: PsychomotorAgitationRequest):
    """Compute multimodal Psychomotor Agitation Index (PAI) from posture, adaptors, and fidgeting."""
    posture = PosturalMetrics(
        posture_state=req.posture_state or PostureState.UPRIGHT.value,
        slump_index=req.slump_index or 0.20,
    )
    cat = SomatosensoryEngine.ADAPTOR_CATEGORIES.get(
        AdaptorType(req.primary_adaptor_type) if req.primary_adaptor_type in [a.value for a in AdaptorType] else AdaptorType.NONE,
        AdaptorCategory.BASELINE_NONE.value
    )
    adaptor = MicroGestureAdaptor(
        adaptor_type=req.primary_adaptor_type or AdaptorType.NONE.value,
        category=cat,
        active=req.primary_adaptor_active or False,
    )
    fidgeting = FidgetingDynamics(
        restlessness_score=req.restlessness_score,
        is_fidgeting=req.is_fidgeting or False,
    )

    agitation = SomatosensoryEngine.fuse_psychomotor_agitation(
        posture=posture,
        primary_adaptor=adaptor,
        fidgeting=fidgeting,
        autonomic_stress=req.autonomic_stress or 0.25,
        cognitive_workload=req.cognitive_workload or 0.30,
        acoustic_jitter=req.acoustic_jitter or 0.02,
    )
    return {"status": "success", "agitation": agitation.to_dict()}


@router.get("/api/somatosensory/config")
async def get_somatosensory_config():
    """Retrieve somatosensory thresholds, posture states, adaptor types, and psychomotor tiers."""
    return {
        "status": "online",
        "slump_threshold": somatosensory_engine.slump_threshold,
        "shoulder_asymmetry_threshold": somatosensory_engine.shoulder_asymmetry_threshold,
        "fidgeting_variance_threshold": somatosensory_engine.fidgeting_variance_threshold,
        "posture_states": [p.value for p in PostureState],
        "adaptor_types": [a.value for a in AdaptorType],
        "adaptor_categories": [c.value for c in AdaptorCategory],
        "psychomotor_tiers": [t.value for t in PsychomotorTier],
    }
