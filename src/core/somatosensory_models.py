"""Phase 12: Somatosensory Kinematics, Postural Ergonomics & Micro-Gesture Kinesics Data Models.

Defines domain contracts for upper-body postural ergonomics, hand-to-face micro-gesture
adaptors (self-touching), kinetic restlessness / fidgeting dynamics, kinesic expressivity,
and the multi-sensor Psychomotor Agitation Index (PAI).
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any


class PostureState(str, Enum):
    """Upper-body postural alignment categories."""
    UPRIGHT = "Upright"
    SLUMPED = "Slumped"
    TENSE_ELEVATED = "Tense_Elevated"
    LATERAL_LEAN = "Lateral_Lean"


class AdaptorType(str, Enum):
    """Hand-to-face micro-gesture / self-touch adaptor categories."""
    NONE = "None"
    CHIN_SUPPORT = "Chin_Support"
    MOUTH_COVER = "Mouth_Cover"
    TEMPLE_RUB = "Temple_Rub"
    EYE_RUB = "Eye_Rub"
    NECK_TOUCH = "Neck_Touch"
    CHEEK_TOUCH = "Cheek_Touch"


class AdaptorCategory(str, Enum):
    """Psychological and affective function of self-touch adaptors."""
    BASELINE_NONE = "Baseline_None"
    EVALUATIVE_COGNITIVE = "Evaluative_Cognitive"
    DEFENSIVE_UNCERTAIN = "Defensive_Uncertain"
    FATIGUE_OVERLOAD = "Fatigue_Overload"
    PACIFYING_STRESS = "Pacifying_Stress"


class PsychomotorTier(str, Enum):
    """Psychomotor agitation and restlessness classification tiers."""
    COMPOSED = "Composed"
    RESTLESS_MILD = "Restless_Mild"
    AGITATED_HIGH = "Agitated_High"
    ACUTE_MOTOR_STORM = "Acute_Motor_Storm"


@dataclass
class PosturalMetrics:
    """Upper-body spinal ergonomics and head-neck alignment telemetry."""
    forward_head_angle_deg: float = 52.0      # Forward head angle relative to vertical (normal: 50-60)
    spinal_tilt_deg: float = 2.5              # Coronal lateral spinal tilt (-20 to +20 deg)
    shoulder_elevation_asymmetry: float = 0.04 # Normalized vertical shoulder asymmetry (0.0 to 1.0)
    slump_index: float = 0.18                 # Normalized postural slump magnitude (0.0 to 1.0)
    posture_state: str = PostureState.UPRIGHT.value
    confidence: float = 0.90
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MicroGestureAdaptor:
    """Hand-to-face micro-gesture adaptor (self-touching) detection."""
    adaptor_type: str = AdaptorType.NONE.value
    category: str = AdaptorCategory.BASELINE_NONE.value
    proximity_distance: float = 0.85          # Normalized proximity to facial region
    active: bool = False
    duration_ms: float = 0.0
    confidence: float = 0.85
    anatomical_region: str = "Face"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FidgetingDynamics:
    """Kinetic energy variance, spatial displacement, and restlessness."""
    kinetic_energy: float = 0.012             # Mean instantaneous kinetic energy
    kinetic_variance: float = 0.008           # Variance across sliding window (high = fidgeting)
    displacement_velocity: float = 14.0       # Spatial movement velocity (px/s or norm/s)
    restlessness_score: float = 0.15          # Normalized restlessness index (0.0 to 1.0)
    fidget_frequency_bpm: float = 12.0        # Rapid micro-motion cycles per minute
    is_fidgeting: bool = False
    state: str = PsychomotorTier.COMPOSED.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KinesicExpressivity:
    """Gesticulation amplitude, spatial volume, and postural expansion."""
    gesture_amplitude: float = 0.35           # Spatial reach of arm movements (0.0 to 1.0)
    expressivity_score: float = 0.42          # Overall kinesic dynamism (0.0 to 1.0)
    expansion_ratio: float = 0.78             # Torso/chest openness ratio (>0.6 = open posture)
    gesticulation_rate: float = 18.0          # Gestures per minute during active speech
    is_open_posture: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PsychomotorAgitationIndex:
    """Composite multi-sensor Psychomotor Agitation Index (0.0 to 1.0)."""
    agitation_index: float = 0.22             # Composite agitation score (0.0 to 1.0)
    tier: str = PsychomotorTier.COMPOSED.value
    psychomotor_slowing: bool = False         # Flags psychomotor retardation / lethargy
    somatic_stress_load: float = 0.20         # Somatic tension burden (0.0 to 1.0)
    contributing_factors: List[str] = field(default_factory=lambda: ["Neutral Posture", "Composed Hands"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SomatosensorySnapshot:
    """Complete somatosensory, posture, and kinesic telemetry snapshot."""
    timestamp: float = 0.0
    posture: PosturalMetrics = field(default_factory=PosturalMetrics)
    adaptors: List[MicroGestureAdaptor] = field(default_factory=list)
    primary_adaptor: MicroGestureAdaptor = field(default_factory=MicroGestureAdaptor)
    fidgeting: FidgetingDynamics = field(default_factory=FidgetingDynamics)
    expressivity: KinesicExpressivity = field(default_factory=KinesicExpressivity)
    agitation: PsychomotorAgitationIndex = field(default_factory=PsychomotorAgitationIndex)
    body_detected: bool = True

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["posture"] = self.posture.to_dict()
        res["adaptors"] = [a.to_dict() for a in self.adaptors]
        res["primary_adaptor"] = self.primary_adaptor.to_dict()
        res["fidgeting"] = self.fidgeting.to_dict()
        res["expressivity"] = self.expressivity.to_dict()
        res["agitation"] = self.agitation.to_dict()
        return res
