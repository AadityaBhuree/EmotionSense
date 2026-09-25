"""Phase 12: Real-Time Somatosensory Kinematics & Postural Ergonomics Engine.

Implements upper-body spinal alignment tracking, forward head posture angle (FHP),
shoulder elevation tension asymmetry, hand-to-face micro-gesture self-touch adaptors
(chin support, mouth cover, temple rub, neck touch), kinetic restlessness / fidgeting
dynamics, kinesic expressivity, and composite Psychomotor Agitation Index (PAI) fusion.
"""

import time
from typing import List, Tuple, Optional, Dict
import numpy as np

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


class SomatosensoryEngine:
    """Enterprise computer vision and somatosensory kinesics intelligence engine."""

    # Proximity thresholds in normalized torso coordinates
    ADAPTOR_PROXIMITY_THRESHOLDS = {
        AdaptorType.CHIN_SUPPORT: 0.16,
        AdaptorType.MOUTH_COVER: 0.14,
        AdaptorType.TEMPLE_RUB: 0.15,
        AdaptorType.EYE_RUB: 0.14,
        AdaptorType.NECK_TOUCH: 0.20,
        AdaptorType.CHEEK_TOUCH: 0.16,
    }

    ADAPTOR_CATEGORIES = {
        AdaptorType.CHIN_SUPPORT: AdaptorCategory.EVALUATIVE_COGNITIVE.value,
        AdaptorType.MOUTH_COVER: AdaptorCategory.DEFENSIVE_UNCERTAIN.value,
        AdaptorType.TEMPLE_RUB: AdaptorCategory.FATIGUE_OVERLOAD.value,
        AdaptorType.EYE_RUB: AdaptorCategory.FATIGUE_OVERLOAD.value,
        AdaptorType.NECK_TOUCH: AdaptorCategory.PACIFYING_STRESS.value,
        AdaptorType.CHEEK_TOUCH: AdaptorCategory.EVALUATIVE_COGNITIVE.value,
        AdaptorType.NONE: AdaptorCategory.BASELINE_NONE.value,
    }

    def __init__(
        self,
        slump_threshold: float = 0.45,
        shoulder_asymmetry_threshold: float = 0.12,
        fidgeting_variance_threshold: float = 0.015,
        buffer_window_sec: float = 30.0,
    ):
        self.slump_threshold = slump_threshold
        self.shoulder_asymmetry_threshold = shoulder_asymmetry_threshold
        self.fidgeting_variance_threshold = fidgeting_variance_threshold
        self.buffer_window_sec = buffer_window_sec

        # Temporal buffers
        self.hand_motion_buffer: List[Tuple[float, float, float]] = []  # (timestamp, left_speed, right_speed)
        self.posture_history: List[Tuple[float, float]] = []            # (timestamp, slump_index)
        self.adaptor_durations: Dict[str, float] = {k.value: 0.0 for k in AdaptorType}
        self.last_adaptor_time: Dict[str, float] = {}

    def compute_postural_metrics(
        self,
        left_shoulder: Tuple[float, float],
        right_shoulder: Tuple[float, float],
        nose: Tuple[float, float],
        left_ear: Optional[Tuple[float, float]] = None,
        right_ear: Optional[Tuple[float, float]] = None,
        timestamp: Optional[float] = None,
    ) -> PosturalMetrics:
        """Calculates upper-body spinal alignment, shoulder asymmetry, and slump index."""
        if timestamp is None:
            timestamp = time.time()

        ls = np.asarray(left_shoulder, dtype=np.float64)
        rs = np.asarray(right_shoulder, dtype=np.float64)
        n = np.asarray(nose, dtype=np.float64)

        torso_width = float(np.linalg.norm(ls - rs))
        if torso_width < 1e-4:
            torso_width = 0.40  # Fallback normalized torso width

        mid_shoulder = (ls + rs) / 2.0

        # Vertical shoulder elevation asymmetry normalized by torso width
        asymmetry = float(abs(ls[1] - rs[1]) / torso_width)

        # Coronal spinal lateral tilt angle (degrees from vertical)
        dx = float(n[0] - mid_shoulder[0])
        dy = float(mid_shoulder[1] - n[1])  # Y grows downwards in image coords
        spinal_tilt = float(np.degrees(np.arctan2(dx, max(1e-4, dy))))

        # Forward head posture (FHP) angle
        if left_ear is not None and right_ear is not None:
            mid_ear = (np.asarray(left_ear) + np.asarray(right_ear)) / 2.0
            ear_shoulder_dx = float(mid_ear[0] - mid_shoulder[0])
            ear_shoulder_dy = float(mid_shoulder[1] - mid_ear[1])
            fhp_angle = float(np.degrees(np.arctan2(max(1e-4, ear_shoulder_dy), abs(ear_shoulder_dx) + 1e-4)))
        else:
            # Fallback estimation based on nose vertical distance to shoulders
            head_drop_ratio = float((mid_shoulder[1] - n[1]) / torso_width)
            fhp_angle = float(np.clip(30.0 + head_drop_ratio * 35.0, 25.0, 75.0))

        # Slump Index calculation (0.0 = perfect upright, 1.0 = heavy forward collapse)
        # Normal vertical ratio from nose to mid-shoulder is roughly 0.70 to 1.10 of torso width
        vert_head_dist = float((mid_shoulder[1] - n[1]) / torso_width)
        slump_raw = float(np.clip(1.0 - ((vert_head_dist - 0.40) / 0.50), 0.0, 1.0))

        self.posture_history.append((timestamp, slump_raw))
        min_time = timestamp - self.buffer_window_sec
        self.posture_history = [(t, s) for t, s in self.posture_history if t >= min_time]

        # Determine postural state
        if asymmetry > self.shoulder_asymmetry_threshold:
            state = PostureState.TENSE_ELEVATED.value
        elif abs(spinal_tilt) > 12.0:
            state = PostureState.LATERAL_LEAN.value
        elif slump_raw >= self.slump_threshold:
            state = PostureState.SLUMPED.value
        else:
            state = PostureState.UPRIGHT.value

        return PosturalMetrics(
            forward_head_angle_deg=round(fhp_angle, 1),
            spinal_tilt_deg=round(spinal_tilt, 1),
            shoulder_elevation_asymmetry=round(asymmetry, 3),
            slump_index=round(slump_raw, 3),
            posture_state=state,
            confidence=0.92,
            is_valid=True,
        )

    def detect_adaptors(
        self,
        face_anchor: Tuple[float, float],
        torso_width: float,
        left_hand: Optional[Tuple[float, float]] = None,
        right_hand: Optional[Tuple[float, float]] = None,
        timestamp: Optional[float] = None,
    ) -> Tuple[List[MicroGestureAdaptor], MicroGestureAdaptor]:
        """Detects hand-to-face micro-gesture self-touch adaptors."""
        if timestamp is None:
            timestamp = time.time()

        if torso_width < 1e-4:
            torso_width = 0.40

        fx, fy = face_anchor
        # Approximate anatomical touch zones relative to face anchor (center of face)
        zones = {
            AdaptorType.CHIN_SUPPORT: (fx, fy + 0.35 * torso_width, "Chin"),
            AdaptorType.MOUTH_COVER: (fx, fy + 0.15 * torso_width, "Mouth"),
            AdaptorType.TEMPLE_RUB: (fx + 0.28 * torso_width, fy - 0.15 * torso_width, "Temple"),
            AdaptorType.EYE_RUB: (fx, fy - 0.10 * torso_width, "Ocular"),
            AdaptorType.NECK_TOUCH: (fx, fy + 0.60 * torso_width, "Neck"),
            AdaptorType.CHEEK_TOUCH: (fx + 0.22 * torso_width, fy + 0.08 * torso_width, "Cheek"),
        }

        detected_adaptors: List[MicroGestureAdaptor] = []
        hands = [h for h in [left_hand, right_hand] if h is not None]

        for adaptor_type, (zx, zy, region) in zones.items():
            thresh = self.ADAPTOR_PROXIMITY_THRESHOLDS[adaptor_type]
            min_dist = 999.0
            for hx, hy in hands:
                # Euclidean distance normalized by torso width
                d = float(np.sqrt((hx - zx) ** 2 + (hy - zy) ** 2) / torso_width)
                if d < min_dist:
                    min_dist = d

            is_active = min_dist <= thresh and len(hands) > 0

            # Update duration
            dur_key = adaptor_type.value
            if is_active:
                last_t = self.last_adaptor_time.get(dur_key, timestamp)
                dt = (timestamp - last_t) * 1000.0
                if dt < 2000.0:
                    self.adaptor_durations[dur_key] += dt
                else:
                    self.adaptor_durations[dur_key] = 200.0
                self.last_adaptor_time[dur_key] = timestamp
            else:
                self.adaptor_durations[dur_key] = 0.0

            if is_active:
                category = self.ADAPTOR_CATEGORIES[adaptor_type]
                adaptor = MicroGestureAdaptor(
                    adaptor_type=adaptor_type.value,
                    category=category,
                    proximity_distance=round(min_dist, 3),
                    active=True,
                    duration_ms=round(self.adaptor_durations[dur_key], 1),
                    confidence=round(float(np.clip(1.0 - (min_dist / max(1e-4, thresh * 1.5)), 0.60, 0.98)), 2),
                    anatomical_region=region,
                )
                detected_adaptors.append(adaptor)

        # Determine primary adaptor
        if detected_adaptors:
            primary = min(detected_adaptors, key=lambda a: a.proximity_distance)
        else:
            primary = MicroGestureAdaptor(
                adaptor_type=AdaptorType.NONE.value,
                category=AdaptorCategory.BASELINE_NONE.value,
                proximity_distance=1.0,
                active=False,
                duration_ms=0.0,
                confidence=0.90,
                anatomical_region="None",
            )

        return detected_adaptors, primary

    def track_fidgeting(
        self,
        left_hand: Optional[Tuple[float, float]],
        right_hand: Optional[Tuple[float, float]],
        timestamp: Optional[float] = None,
    ) -> FidgetingDynamics:
        """Tracks kinetic displacement, energy variance, and restless fidgeting."""
        if timestamp is None:
            timestamp = time.time()

        lh_speed = 0.0
        rh_speed = 0.0

        if len(self.hand_motion_buffer) > 0:
            last_t, last_lh_x, last_rh_x = self.hand_motion_buffer[-1]
            dt = max(0.001, timestamp - last_t)

            if left_hand is not None:
                lh_speed = float(abs(left_hand[0] - last_lh_x) / dt)
            if right_hand is not None:
                rh_speed = float(abs(right_hand[0] - last_rh_x) / dt)

        cur_lh_x = left_hand[0] if left_hand is not None else 0.0
        cur_rh_x = right_hand[0] if right_hand is not None else 0.0
        self.hand_motion_buffer.append((timestamp, cur_lh_x, cur_rh_x))

        # Evict outside buffer window
        min_time = timestamp - self.buffer_window_sec
        self.hand_motion_buffer = [(t, lx, rx) for t, lx, rx in self.hand_motion_buffer if t >= min_time]

        mean_speed = float((lh_speed + rh_speed) / 2.0)
        kinetic_energy = float(0.5 * (mean_speed ** 2))

        # Compute kinetic variance across temporal window
        if len(self.hand_motion_buffer) >= 4:
            # Estimate velocities across buffer
            vels = []
            for i in range(1, len(self.hand_motion_buffer)):
                dt_i = max(0.001, self.hand_motion_buffer[i][0] - self.hand_motion_buffer[i - 1][0])
                v = abs(self.hand_motion_buffer[i][1] - self.hand_motion_buffer[i - 1][1]) / dt_i
                vels.append(v)
            kinetic_var = float(np.var(vels)) if vels else 0.005
        else:
            kinetic_var = 0.005

        # Restlessness score (0.0 to 1.0)
        restlessness = float(np.clip((kinetic_var / (self.fidgeting_variance_threshold * 3.0)) + (kinetic_energy * 2.0), 0.05, 0.98))
        fidget_bpm = float(np.clip(restlessness * 45.0, 5.0, 60.0))

        is_fidgeting = kinetic_var >= self.fidgeting_variance_threshold or restlessness >= 0.50

        # Tier assignment
        if restlessness < 0.25:
            tier = PsychomotorTier.COMPOSED.value
        elif restlessness < 0.55:
            tier = PsychomotorTier.RESTLESS_MILD.value
        elif restlessness < 0.80:
            tier = PsychomotorTier.AGITATED_HIGH.value
        else:
            tier = PsychomotorTier.ACUTE_MOTOR_STORM.value

        return FidgetingDynamics(
            kinetic_energy=round(kinetic_energy, 4),
            kinetic_variance=round(kinetic_var, 4),
            displacement_velocity=round(mean_speed, 2),
            restlessness_score=round(restlessness, 3),
            fidget_frequency_bpm=round(fidget_bpm, 1),
            is_fidgeting=is_fidgeting,
            state=tier,
        )

    @staticmethod
    def compute_kinesic_expressivity(
        torso_width: float,
        left_hand: Optional[Tuple[float, float]] = None,
        right_hand: Optional[Tuple[float, float]] = None,
        speech_active: bool = True,
    ) -> KinesicExpressivity:
        """Quantifies gesticulation spatial amplitude, chest openness, and expressivity."""
        if torso_width < 1e-4:
            torso_width = 0.40

        if left_hand is not None and right_hand is not None:
            hand_span = float(np.linalg.norm(np.asarray(left_hand) - np.asarray(right_hand)))
            expansion_ratio = float(np.clip(hand_span / (torso_width * 1.8), 0.10, 1.20))
            is_open = expansion_ratio >= 0.50
            amplitude = float(np.clip(hand_span / (torso_width * 2.5), 0.05, 1.0))
        else:
            expansion_ratio = 0.75
            is_open = True
            amplitude = 0.30

        # Expressivity score influenced by speech turn
        speech_mult = 1.2 if speech_active else 0.8
        expressivity = float(np.clip(amplitude * 0.6 + (expansion_ratio * 0.4) * speech_mult, 0.05, 0.98))
        gest_rate = float(np.clip(expressivity * 35.0, 0.0, 50.0))

        return KinesicExpressivity(
            gesture_amplitude=round(amplitude, 3),
            expressivity_score=round(expressivity, 3),
            expansion_ratio=round(expansion_ratio, 3),
            gesticulation_rate=round(gest_rate, 1),
            is_open_posture=is_open,
        )

    @staticmethod
    def fuse_psychomotor_agitation(
        posture: PosturalMetrics,
        primary_adaptor: MicroGestureAdaptor,
        fidgeting: FidgetingDynamics,
        autonomic_stress: float = 0.25,
        cognitive_workload: float = 0.30,
        acoustic_jitter: float = 0.02,
    ) -> PsychomotorAgitationIndex:
        """Multimodal late fusion of posture, micro-gestures, fidgeting, and autonomic strain."""
        # Somatic tension burden (0.0 to 1.0)
        tension_from_posture = 0.35 if posture.posture_state == PostureState.TENSE_ELEVATED.value else (posture.slump_index * 0.4)
        adaptor_penalty = 0.25 if primary_adaptor.active and primary_adaptor.category == AdaptorCategory.PACIFYING_STRESS.value else (0.15 if primary_adaptor.active else 0.0)

        somatic_stress = float(np.clip(
            0.30 * tension_from_posture +
            0.25 * adaptor_penalty +
            0.25 * autonomic_stress +
            0.20 * cognitive_workload,
            0.05, 0.98
        ))

        # Composite Psychomotor Agitation Index (0.0 to 1.0)
        pai_raw = (
            0.35 * fidgeting.restlessness_score +
            0.20 * somatic_stress +
            0.15 * adaptor_penalty +
            0.15 * autonomic_stress +
            0.15 * float(np.clip(acoustic_jitter * 15.0, 0.0, 1.0))
        )
        pai_score = float(np.clip(pai_raw, 0.05, 0.98))

        # Detect psychomotor slowing (retardation): low fidgeting, slumped posture, low tension
        psychomotor_slowing = (
            fidgeting.restlessness_score < 0.12 and
            posture.slump_index >= 0.50 and
            not primary_adaptor.active
        )

        # Tier assignment
        if pai_score < 0.25:
            tier = PsychomotorTier.COMPOSED.value
        elif pai_score < 0.55:
            tier = PsychomotorTier.RESTLESS_MILD.value
        elif pai_score < 0.80:
            tier = PsychomotorTier.AGITATED_HIGH.value
        else:
            tier = PsychomotorTier.ACUTE_MOTOR_STORM.value

        factors = []
        if fidgeting.is_fidgeting:
            factors.append("Kinetic Restlessness / Fidgeting")
        if primary_adaptor.active:
            factors.append(f"Active {primary_adaptor.adaptor_type.replace('_', ' ')} Self-Touch")
        if posture.posture_state == PostureState.TENSE_ELEVATED.value:
            factors.append("Elevated Trapezius / Shoulder Tension")
        elif posture.posture_state == PostureState.SLUMPED.value:
            factors.append("Postural Slump / Spinal Flexion")
        if autonomic_stress > 0.55:
            factors.append("Autonomic Cardiac Stress Strain")
        if psychomotor_slowing:
            factors.append("Psychomotor Retardation / Kinetic Lethargy")
        if not factors:
            factors = ["Ergonomic Spine Alignment", "Stable Composure"]

        return PsychomotorAgitationIndex(
            agitation_index=round(pai_score, 3),
            tier=tier,
            psychomotor_slowing=psychomotor_slowing,
            somatic_stress_load=round(somatic_stress, 3),
            contributing_factors=factors[:3],
        )

    def process_frame(
        self,
        left_shoulder: Tuple[float, float],
        right_shoulder: Tuple[float, float],
        nose: Tuple[float, float],
        left_ear: Optional[Tuple[float, float]] = None,
        right_ear: Optional[Tuple[float, float]] = None,
        left_hand: Optional[Tuple[float, float]] = None,
        right_hand: Optional[Tuple[float, float]] = None,
        autonomic_stress: float = 0.25,
        cognitive_workload: float = 0.30,
        acoustic_jitter: float = 0.02,
        speech_active: bool = True,
        timestamp: Optional[float] = None,
    ) -> SomatosensorySnapshot:
        """Processes a single video frame's upper-body keypoints into a full SomatosensorySnapshot."""
        if timestamp is None:
            timestamp = time.time()

        posture = self.compute_postural_metrics(
            left_shoulder, right_shoulder, nose, left_ear, right_ear, timestamp=timestamp
        )

        torso_w = float(np.linalg.norm(np.asarray(left_shoulder) - np.asarray(right_shoulder)))
        adaptors, primary_adaptor = self.detect_adaptors(
            face_anchor=nose,
            torso_width=torso_w,
            left_hand=left_hand,
            right_hand=right_hand,
            timestamp=timestamp,
        )

        fidgeting = self.track_fidgeting(left_hand, right_hand, timestamp=timestamp)
        expressivity = self.compute_kinesic_expressivity(
            torso_width=torso_w,
            left_hand=left_hand,
            right_hand=right_hand,
            speech_active=speech_active,
        )

        agitation = self.fuse_psychomotor_agitation(
            posture=posture,
            primary_adaptor=primary_adaptor,
            fidgeting=fidgeting,
            autonomic_stress=autonomic_stress,
            cognitive_workload=cognitive_workload,
            acoustic_jitter=acoustic_jitter,
        )

        return SomatosensorySnapshot(
            timestamp=timestamp,
            posture=posture,
            adaptors=adaptors,
            primary_adaptor=primary_adaptor,
            fidgeting=fidgeting,
            expressivity=expressivity,
            agitation=agitation,
            body_detected=True,
        )
