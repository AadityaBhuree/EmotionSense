"""Unit tests for Phase 12 Somatosensory Kinematics, Posture Ergonomics & Adaptor Engine."""

import time
import pytest

from src.core.somatosensory_models import (
    PostureState,
    AdaptorType,
    AdaptorCategory,
    PsychomotorTier,
    PosturalMetrics,
    MicroGestureAdaptor,
    FidgetingDynamics,
    KinesicExpressivity,
    PsychomotorAgitationIndex,
    SomatosensorySnapshot,
)
from src.analytics.somatosensory import SomatosensoryEngine


class TestSomatosensoryModels:
    """Tests dataclass integrity and dictionary serialization for somatosensory models."""

    def test_postural_metrics_defaults(self):
        pm = PosturalMetrics()
        d = pm.to_dict()
        assert d["posture_state"] == PostureState.UPRIGHT.value
        assert d["is_valid"] is True
        assert 0.0 <= d["slump_index"] <= 1.0

    def test_micro_gesture_adaptor_defaults(self):
        mga = MicroGestureAdaptor()
        d = mga.to_dict()
        assert d["adaptor_type"] == AdaptorType.NONE.value
        assert d["category"] == AdaptorCategory.BASELINE_NONE.value
        assert d["active"] is False

    def test_fidgeting_dynamics_defaults(self):
        fd = FidgetingDynamics()
        d = fd.to_dict()
        assert d["state"] == PsychomotorTier.COMPOSED.value
        assert d["is_fidgeting"] is False
        assert 0.0 <= d["restlessness_score"] <= 1.0

    def test_kinesic_expressivity_defaults(self):
        ke = KinesicExpressivity()
        d = ke.to_dict()
        assert d["is_open_posture"] is True
        assert 0.0 <= d["expressivity_score"] <= 1.0

    def test_psychomotor_agitation_index_defaults(self):
        pai = PsychomotorAgitationIndex()
        d = pai.to_dict()
        assert d["tier"] == PsychomotorTier.COMPOSED.value
        assert 0.0 <= d["agitation_index"] <= 1.0
        assert isinstance(d["contributing_factors"], list)

    def test_somatosensory_snapshot_serialization(self):
        snap = SomatosensorySnapshot()
        d = snap.to_dict()
        assert "posture" in d
        assert "primary_adaptor" in d
        assert "fidgeting" in d
        assert "agitation" in d
        assert d["body_detected"] is True


class TestSomatosensoryEngine:
    """Tests core kinematic algorithms, posture, adaptors, and psychomotor fusion."""

    @pytest.fixture
    def engine(self):
        return SomatosensoryEngine()

    def test_compute_postural_metrics_upright(self, engine):
        # Symmetrical shoulders with nose high above shoulders
        ls = (0.30, 0.60)
        rs = (0.70, 0.60)
        nose = (0.50, 0.25)
        pm = engine.compute_postural_metrics(ls, rs, nose)

        assert pm.posture_state == PostureState.UPRIGHT.value
        assert pm.slump_index < 0.35
        assert pm.shoulder_elevation_asymmetry < 0.05
        assert abs(pm.spinal_tilt_deg) < 5.0

    def test_compute_postural_metrics_slumped(self, engine):
        # Nose dropped down very close to shoulder level
        ls = (0.30, 0.60)
        rs = (0.70, 0.60)
        nose = (0.50, 0.52)  # Low head = slumped
        pm = engine.compute_postural_metrics(ls, rs, nose)

        assert pm.posture_state == PostureState.SLUMPED.value
        assert pm.slump_index >= 0.45

    def test_compute_postural_metrics_tense_elevated(self, engine):
        # One shoulder significantly higher than the other (vertical asymmetry)
        ls = (0.30, 0.48)
        rs = (0.70, 0.60)
        nose = (0.50, 0.25)
        pm = engine.compute_postural_metrics(ls, rs, nose)

        assert pm.posture_state == PostureState.TENSE_ELEVATED.value
        assert pm.shoulder_elevation_asymmetry > 0.12

    def test_compute_postural_metrics_lateral_lean(self, engine):
        # Lateral head shift relative to shoulders
        ls = (0.30, 0.60)
        rs = (0.70, 0.60)
        nose = (0.75, 0.25)  # Heavy tilt to right
        pm = engine.compute_postural_metrics(ls, rs, nose)

        assert pm.posture_state == PostureState.LATERAL_LEAN.value
        assert abs(pm.spinal_tilt_deg) > 12.0

    def test_detect_adaptors_none_active(self, engine):
        face_anchor = (0.50, 0.30)
        torso_w = 0.40
        # Hands resting low at lap (y = 0.85)
        lh = (0.40, 0.85)
        rh = (0.60, 0.85)
        adaptors, primary = engine.detect_adaptors(face_anchor, torso_w, lh, rh)

        assert len(adaptors) == 0
        assert primary.adaptor_type == AdaptorType.NONE.value
        assert primary.active is False

    def test_detect_adaptors_chin_support(self, engine):
        face_anchor = (0.50, 0.30)
        torso_w = 0.40
        # Chin zone is roughly (0.50, 0.30 + 0.35 * 0.40) = (0.50, 0.44)
        lh = (0.50, 0.43)
        adaptors, primary = engine.detect_adaptors(face_anchor, torso_w, left_hand=lh)

        assert any(a.adaptor_type == AdaptorType.CHIN_SUPPORT.value for a in adaptors)
        assert primary.adaptor_type == AdaptorType.CHIN_SUPPORT.value
        assert primary.category == AdaptorCategory.EVALUATIVE_COGNITIVE.value
        assert primary.active is True

    def test_detect_adaptors_mouth_cover(self, engine):
        face_anchor = (0.50, 0.30)
        torso_w = 0.40
        # Mouth zone: (0.50, 0.30 + 0.15 * 0.40) = (0.50, 0.36)
        rh = (0.50, 0.35)
        adaptors, primary = engine.detect_adaptors(face_anchor, torso_w, right_hand=rh)

        assert any(a.adaptor_type == AdaptorType.MOUTH_COVER.value for a in adaptors)
        assert primary.category == AdaptorCategory.DEFENSIVE_UNCERTAIN.value

    def test_track_fidgeting_calm(self, engine):
        # Static hands
        t0 = time.time()
        for i in range(5):
            fd = engine.track_fidgeting((0.40, 0.80), (0.60, 0.80), timestamp=t0 + i * 0.1)

        assert fd.state in [PsychomotorTier.COMPOSED.value, PsychomotorTier.RESTLESS_MILD.value]
        assert fd.is_fidgeting is False

    def test_track_fidgeting_rapid_motion(self, engine):
        # Rapid erratic hand shifts
        t0 = time.time()
        for i in range(10):
            x_shift = 0.15 if i % 2 == 0 else -0.15
            fd = engine.track_fidgeting((0.40 + x_shift, 0.80), (0.60 - x_shift, 0.80), timestamp=t0 + i * 0.05)

        assert fd.is_fidgeting is True
        assert fd.restlessness_score > 0.40

    def test_compute_kinesic_expressivity(self):
        ke_wide = SomatosensoryEngine.compute_kinesic_expressivity(
            torso_width=0.40,
            left_hand=(0.10, 0.50),
            right_hand=(0.90, 0.50),
            speech_active=True,
        )
        assert ke_wide.is_open_posture is True
        assert ke_wide.expressivity_score > 0.40

        ke_closed = SomatosensoryEngine.compute_kinesic_expressivity(
            torso_width=0.40,
            left_hand=(0.48, 0.55),
            right_hand=(0.52, 0.55),
            speech_active=False,
        )
        assert ke_closed.expansion_ratio < 0.50
        assert ke_closed.is_open_posture is False

    def test_fuse_psychomotor_agitation(self):
        pm = PosturalMetrics(posture_state=PostureState.UPRIGHT.value, slump_index=0.15)
        adaptor = MicroGestureAdaptor(active=False)
        fidgeting = FidgetingDynamics(restlessness_score=0.10, is_fidgeting=False)

        pai_calm = SomatosensoryEngine.fuse_psychomotor_agitation(
            posture=pm,
            primary_adaptor=adaptor,
            fidgeting=fidgeting,
            autonomic_stress=0.20,
        )
        assert pai_calm.tier == PsychomotorTier.COMPOSED.value
        assert pai_calm.agitation_index < 0.30

        # Highly agitated scenario
        pm_tense = PosturalMetrics(posture_state=PostureState.TENSE_ELEVATED.value, slump_index=0.40)
        adaptor_neck = MicroGestureAdaptor(
            adaptor_type=AdaptorType.NECK_TOUCH.value,
            category=AdaptorCategory.PACIFYING_STRESS.value,
            active=True,
        )
        fidgeting_high = FidgetingDynamics(restlessness_score=0.85, is_fidgeting=True)

        pai_high = SomatosensoryEngine.fuse_psychomotor_agitation(
            posture=pm_tense,
            primary_adaptor=adaptor_neck,
            fidgeting=fidgeting_high,
            autonomic_stress=0.80,
        )
        assert pai_high.tier in [PsychomotorTier.AGITATED_HIGH.value, PsychomotorTier.ACUTE_MOTOR_STORM.value]
        assert pai_high.agitation_index > 0.50

    def test_process_frame_end_to_end(self, engine):
        snap = engine.process_frame(
            left_shoulder=(0.30, 0.60),
            right_shoulder=(0.70, 0.60),
            nose=(0.50, 0.30),
            left_ear=(0.40, 0.28),
            right_ear=(0.60, 0.28),
            left_hand=(0.50, 0.44),  # Chin support
            right_hand=(0.65, 0.85),
            autonomic_stress=0.35,
            cognitive_workload=0.40,
        )
        assert isinstance(snap, SomatosensorySnapshot)
        assert snap.body_detected is True
        assert snap.posture.posture_state == PostureState.UPRIGHT.value
        assert snap.primary_adaptor.adaptor_type == AdaptorType.CHIN_SUPPORT.value
        assert snap.agitation.tier in [PsychomotorTier.COMPOSED.value, PsychomotorTier.RESTLESS_MILD.value]
