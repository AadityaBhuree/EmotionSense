"""Core models, configurations, and domain data contracts."""

from src.core.types import (
    EmotionCategory,
    FacialActionUnits,
    VisionEmotionResult,
    AcousticFeatures,
    VoiceEmotionResult,
    AffectVector,
    MultimodalEmotionState,
    SessionRecord,
    AcousticSERResult,
    CrossModalAttentionWeights,
    ModalityCongruence,
    CrossModalFusionResult,
    LongitudinalSessionPoint,
    AffectiveDriftMetrics,
    CohortBenchmark,
    LongitudinalProfile,
    EdgeExecutionProvider,
    QuantizationPrecision,
    EdgeDeviceProfile,
    InferenceBenchmarkResult,
    ModelQuantizationSummary,
    EdgeModelManifest,
)
from src.core.config import (
    EDGE_RUNTIME_CONFIG,
)

from src.core.biometric_models import (
    StressClassification,
    PulseMeasurement,
    HRVMetrics,
    RespirationMetrics,
    AutonomicStressRecord,
    BiometricTelemetry,
)

__all__ = [
    "EmotionCategory",
    "FacialActionUnits",
    "VisionEmotionResult",
    "AcousticFeatures",
    "VoiceEmotionResult",
    "AffectVector",
    "MultimodalEmotionState",
    "SessionRecord",
    "AcousticSERResult",
    "CrossModalAttentionWeights",
    "ModalityCongruence",
    "CrossModalFusionResult",
    "LongitudinalSessionPoint",
    "AffectiveDriftMetrics",
    "CohortBenchmark",
    "LongitudinalProfile",
    "EdgeExecutionProvider",
    "QuantizationPrecision",
    "EdgeDeviceProfile",
    "InferenceBenchmarkResult",
    "ModelQuantizationSummary",
    "EdgeModelManifest",
    "EDGE_RUNTIME_CONFIG",
    "StressClassification",
    "PulseMeasurement",
    "HRVMetrics",
    "RespirationMetrics",
    "AutonomicStressRecord",
    "BiometricTelemetry",
]


