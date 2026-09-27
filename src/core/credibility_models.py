"""Phase 13: Forensic Veracity Assessment & Affective Deception Leakage Data Models.

Defines domain contracts for micro-momentary affective leakage, Duchenne smile
incongruence, 8-14 Hz vocal fold micro-tremor (Voice Stress Analysis), polygraphic
stress cross-checking, and the composite Credibility & Deception Risk Index (CDRI).
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any


class VeracityTier(str, Enum):
    """Forensic veracity and credibility classification tiers."""
    VERIDICAL_AUTHENTIC = "Veridical_Authentic"         # CDRI < 0.25: High congruence, organic affect
    COGNITIVE_STRAIN = "Cognitive_Strain"               # CDRI 0.25 - 0.50: Mental effort, minor tension
    SUSPICIOUS_INCONGRUENCE = "Suspicious_Incongruence" # CDRI 0.50 - 0.75: Observable masking/leakage
    HIGH_DECEPTION_RISK = "High_Deception_Risk"         # CDRI >= 0.75: Multimodal deceit markers aligned


class DeceptionFlag(str, Enum):
    """Categorical deception anomaly flags detected across channels."""
    NON_DUCHENNE_MASKING = "Non_Duchenne_Masking"
    MICRO_FLASH_LEAKAGE = "Micro_Flash_Leakage"
    FACIAL_ASYMMETRY_SNEER = "Facial_Asymmetry_Sneer"
    ACOUSTIC_MICRO_TREMOR = "Acoustic_Micro_Tremor"
    PUPIL_DILATION_STRAIN = "Pupil_Dilation_Strain"
    PACIFYING_ADAPTOR_SURGE = "Pacifying_Adaptor_Surge"
    AUTONOMIC_PULSE_SURGE = "Autonomic_Pulse_Surge"
    LATENCY_ELONGATION = "Latency_Elongation"


@dataclass
class MicroLeakageEvent:
    """Transient sub-200ms micro-expression breakthrough conflicting with macro-affect."""
    timestamp: float = 0.0
    duration_ms: float = 120.0
    leaked_affect: str = "Fear"
    masked_affect: str = "Joy"
    conflicting_action_units: List[str] = field(default_factory=lambda: ["AU04", "AU20"])
    leakage_intensity: float = 0.72
    confidence: float = 0.88

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceStressProfile:
    """Acoustic voice stress analysis (VSA), 8-14 Hz micro-tremor & glottal dynamics."""
    micro_tremor_energy: float = 0.042        # 8-14 Hz Lippold micro-tremor power
    cpp_db: float = 12.5                      # Cepstral Peak Prominence in dB (lower = strained/turbulent)
    f0_perturbation_quotient: float = 0.018    # Pitch perturbation quotient (PPQ)
    spectral_slope: float = -14.2              # Spectral energy slope dB/octave
    response_latency_sec: float = 0.45        # Response hesitation / turn latency
    stress_index: float = 0.18                # Normalized acoustic voice stress (0.0 to 1.0)
    is_voice_stressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FacialVeracityMetrics:
    """Facial FACS action unit congruence, Duchenne genuineness, and sneer asymmetry."""
    duchenne_congruence: float = 0.85         # 1.0 = authentic genuine smile, 0.0 = forced
    duchenne_incongruence_index: float = 0.15 # 0.0 = genuine, 1.0 = social masking (AU12 without AU06)
    sneer_asymmetry_index: float = 0.06       # Lateral unilateral smirk asymmetry (0.0 to 1.0)
    micro_leakage_detected: bool = False
    leakage_events_count: int = 0
    macro_masked_state: str = "Neutral"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultimodalPolygraphProfile:
    """Cross-modal polygraphic channel decomposition for deception cross-checking."""
    facial_incongruence_score: float = 0.15   # Weight: 25%
    voice_stress_score: float = 0.18          # Weight: 25%
    pupil_dilation_strain: float = 0.12       # Weight: 20%
    pacifying_adaptor_score: float = 0.08     # Weight: 15%
    autonomic_pulse_surge_score: float = 0.10 # Weight: 15%
    channel_contributions: Dict[str, float] = field(
        default_factory=lambda: {
            "facial": 0.15,
            "vocal": 0.18,
            "pupillometric": 0.12,
            "somatosensory": 0.08,
            "autonomic": 0.10,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CredibilitySnapshot:
    """Complete forensic veracity, micro-leakage, and credibility risk assessment."""
    timestamp: float = 0.0
    credibility_score: float = 0.86           # 0.0 (high deception) to 1.0 (veridical)
    deception_risk_index: float = 0.14        # 0.0 (truthful) to 1.0 (deceitful)
    tier: str = VeracityTier.VERIDICAL_AUTHENTIC.value
    facial_veracity: FacialVeracityMetrics = field(default_factory=FacialVeracityMetrics)
    voice_stress: VoiceStressProfile = field(default_factory=VoiceStressProfile)
    polygraph: MultimodalPolygraphProfile = field(default_factory=MultimodalPolygraphProfile)
    recent_leakages: List[MicroLeakageEvent] = field(default_factory=list)
    active_flags: List[str] = field(default_factory=list)
    clinical_verdict: str = "High affective congruence; organic vocal prosody and authentic facial valence."

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["facial_veracity"] = self.facial_veracity.to_dict()
        res["voice_stress"] = self.voice_stress.to_dict()
        res["polygraph"] = self.polygraph.to_dict()
        res["recent_leakages"] = [e.to_dict() for e in self.recent_leakages]
        return res
