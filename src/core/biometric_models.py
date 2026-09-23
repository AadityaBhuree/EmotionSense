"""Phase 10: Remote Biometric & Physiological Telemetry Data Models.

Defines domain contracts for contact-free optical photoplethysmography (rPPG),
cardiac pulse rates, Heart Rate Variability (HRV), respiratory sinus arrhythmia (RSA),
and multimodal autonomic stress index calculation.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any


class StressClassification(str, Enum):
    """Autonomic nervous system arousal classification."""
    RELAXED = "Relaxed"
    OPTIMAL_ALERTNESS = "Optimal_Alertness"
    ELEVATED_STRAIN = "Elevated_Strain"
    ACUTE_DISTRESS = "Acute_Distress"


@dataclass
class PulseMeasurement:
    """Instantaneous optical pulse and Blood Volume Pulse (BVP) metric."""
    bpm: float = 72.0
    confidence: float = 0.85
    signal_quality_snr: float = 12.5       # Signal-to-noise ratio in dB
    timestamp: float = 0.0
    bvp_sample: float = 0.0
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HRVMetrics:
    """Time-domain and autonomic Heart Rate Variability (HRV) metrics."""
    sdnn_ms: float = 50.0                  # Standard deviation of NN intervals (ms)
    rmssd_ms: float = 42.0                 # Root Mean Square of Successive Differences (ms)
    pnn50_pct: float = 18.0                # Percentage of successive RR differences > 50ms
    mean_rr_ms: float = 833.0              # Mean inter-beat interval in ms
    baevsky_stress_index: float = 85.0     # Baevsky Stress Index (SI = AMo / (2 * Mo * MxDMn))
    hrv_score: float = 75.0                # Composite vagal tone vitality score (0.0 to 100.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RespirationMetrics:
    """Respiratory rate derived from rPPG amplitude/frequency modulation (RSA)."""
    rpm: float = 15.0                      # Breaths per minute (RPM)
    confidence: float = 0.80
    method: str = "RSA_Modulation"         # RSA_Modulation or Optical_Motion

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AutonomicStressRecord:
    """Composite neuro-physiological and somatic stress assessment."""
    stress_index: float = 0.28             # Normalized stress score (0.0 to 1.0)
    classification: str = StressClassification.OPTIMAL_ALERTNESS.value
    sympathetic_tone: float = 0.35         # Sympathetic fight-or-flight activation (0.0 to 1.0)
    parasympathetic_tone: float = 0.65     # Parasympathetic rest-and-digest vagal tone (0.0 to 1.0)
    somatic_arousal: float = 0.30          # Combined somatic arousal marker
    contributing_factors: List[str] = field(default_factory=lambda: ["Normal Cardiac Rhythm", "Balanced Autonomic Tone"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BiometricTelemetry:
    """Comprehensive real-time biometric telemetry snapshot for a frame or window."""
    timestamp: float = 0.0
    pulse: PulseMeasurement = field(default_factory=PulseMeasurement)
    hrv: HRVMetrics = field(default_factory=HRVMetrics)
    respiration: RespirationMetrics = field(default_factory=RespirationMetrics)
    autonomic_stress: AutonomicStressRecord = field(default_factory=AutonomicStressRecord)
    bvp_history: List[float] = field(default_factory=list)
    rr_intervals_ms: List[float] = field(default_factory=list)
    roi_detected: bool = True

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["pulse"] = self.pulse.to_dict()
        res["hrv"] = self.hrv.to_dict()
        res["respiration"] = self.respiration.to_dict()
        res["autonomic_stress"] = self.autonomic_stress.to_dict()
        return res
