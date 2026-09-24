"""Phase 11: Cognitive Workload & Oculomotor Telemetry Data Models.

Defines domain contracts for pupillometry (Cognitive Pupillary Response - CPR),
eye blink dynamics, PERCLOS drowsiness evaluation, gaze fixation / saccades,
and multi-sensor NASA-TLX Cognitive Workload estimation.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any


class WorkloadTier(str, Enum):
    """Cognitive workload and mental demand classification tiers."""
    LOW_LOAD = "Low_Load"
    OPTIMAL_ENGAGEMENT = "Optimal_Engagement"
    HIGH_EFFORT = "High_Effort"
    COGNITIVE_OVERLOAD = "Cognitive_Overload"


@dataclass
class PupillometryMetrics:
    """Instantaneous pupil dilation and task-evoked cognitive response."""
    pupil_diameter_ratio: float = 0.42       # Pupil-to-Iris Ratio (PIR)
    baseline_ratio: float = 0.40             # Calibrated resting baseline PIR
    dilation_change_pct: float = 5.0         # Relative dilation % from baseline
    cognitive_pupillary_response: float = 0.25 # Normalized CPR index (0.0 to 1.0)
    confidence: float = 0.88
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BlinkDynamics:
    """Eye blink frequency, duration, PERCLOS and drowsiness markers."""
    ear: float = 0.28                        # Eye Aspect Ratio
    blink_rate_bpm: float = 16.0             # Blinks per minute (normal: 12-20)
    mean_blink_duration_ms: float = 220.0    # Average closure duration in ms
    perclos: float = 0.06                    # Proportion of time eye is >80% closed (0.0 to 1.0)
    micro_sleep_detected: bool = False       # Sustained closure > 500ms
    blink_suppressed: bool = False           # Inhibit blinks (<8 bpm) under intense focus

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GazeTelemetry:
    """Gaze vector orientation, fixation stability and saccadic velocity."""
    gaze_yaw_deg: float = 0.0                # Horizontal gaze angle (-45 to +45)
    gaze_pitch_deg: float = 0.0              # Vertical gaze angle (-30 to +30)
    screen_x: float = 0.5                    # Estimated normalized focal screen X (0.0 to 1.0)
    screen_y: float = 0.5                    # Estimated normalized focal screen Y (0.0 to 1.0)
    fixation_duration_ms: float = 350.0      # Current fixation dwell duration
    saccade_velocity_deg_s: float = 120.0    # Saccadic peak angular velocity
    is_saccade: bool = False                 # Active rapid eye movement flag
    dispersion_area: float = 0.12            # Gaze spatial dispersion radius

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NASATLXDimensions:
    """Estimated NASA-TLX dimensional workload sub-scales (0 to 100)."""
    mental_demand: float = 45.0
    temporal_demand: float = 40.0
    effort: float = 50.0
    frustration: float = 25.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CognitiveWorkloadRecord:
    """Composite neuro-ergonomic mental workload evaluation."""
    workload_index: float = 0.38             # Unified workload score (0.0 to 1.0)
    tier: str = WorkloadTier.OPTIMAL_ENGAGEMENT.value
    nasa_tlx: NASATLXDimensions = field(default_factory=NASATLXDimensions)
    mental_exhaustion_risk: float = 0.22     # Cumulative fatigue index (0.0 to 1.0)
    contributing_factors: List[str] = field(default_factory=lambda: ["Balanced Pupillary Tone", "Stable Fixation"])

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["nasa_tlx"] = self.nasa_tlx.to_dict()
        return res


@dataclass
class OculomotorSnapshot:
    """Complete oculomotor and cognitive telemetry frame snapshot."""
    timestamp: float = 0.0
    pupillometry: PupillometryMetrics = field(default_factory=PupillometryMetrics)
    blinks: BlinkDynamics = field(default_factory=BlinkDynamics)
    gaze: GazeTelemetry = field(default_factory=GazeTelemetry)
    workload: CognitiveWorkloadRecord = field(default_factory=CognitiveWorkloadRecord)
    eyes_detected: bool = True

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["pupillometry"] = self.pupillometry.to_dict()
        res["blinks"] = self.blinks.to_dict()
        res["gaze"] = self.gaze.to_dict()
        res["workload"] = self.workload.to_dict()
        return res
