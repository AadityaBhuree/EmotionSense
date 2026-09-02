"""Multimodal fusion and behavioral affect intelligence module."""

from src.fusion.metrics import AffectMetricsCalculator
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.fusion.anomaly_detector import (
    AffectiveAnomalyDetector,
    AffectiveAnomaly,
    AnomalySeverity,
    AnomalyType,
)

__all__ = [
    "AffectMetricsCalculator",
    "MultimodalFusionEngine",
    "AffectiveAnomalyDetector",
    "AffectiveAnomaly",
    "AnomalySeverity",
    "AnomalyType",
]

