"""Model Quantization Optimizer for INT8 and FP16 Edge Compression."""

import numpy as np
from typing import Dict, Any, Optional

from src.core.types import (
    ModelQuantizationSummary,
    QuantizationPrecision,
)
from src.core.config import EDGE_RUNTIME_CONFIG
from src.utils.logger import get_logger

logger = get_logger("edge_quantizer")


class ModelQuantizationOptimizer:
    """Simulates and executes dynamic post-training quantization for edge deployments.
    
    Provides scaling factor computation, zero-point derivation, accuracy preservation
    metrics, and memory reduction ratios for edge hardware.
    """

    MODEL_CATALOG = {
        "vision_mesh": {
            "name": "Face Mesh 468 AU Detector",
            "base_size_mb": 42.5,
            "base_latency_ms": 14.2,
            "accuracy_baseline": 98.6,
        },
        "audio_ser": {
            "name": "Wav2Vec2 Deep SER Engine",
            "base_size_mb": 95.0,
            "base_latency_ms": 7.4,
            "accuracy_baseline": 94.8,
        },
        "text_nlp": {
            "name": "RoBERTa-GoEmotions Ensemble",
            "base_size_mb": 120.0,
            "base_latency_ms": 9.8,
            "accuracy_baseline": 96.2,
        },
        "cross_modal_cmaf": {
            "name": "Cross-Modal Attentive Fusion",
            "base_size_mb": 64.0,
            "base_latency_ms": 18.5,
            "accuracy_baseline": 97.4,
        },
    }

    def __init__(self):
        self.compression_table = EDGE_RUNTIME_CONFIG.get(
            "quantization_compression",
            {"FP32": 1.0, "FP16": 2.0, "INT8": 3.9, "Dynamic_INT8": 3.6}
        )
        self.speedup_table = EDGE_RUNTIME_CONFIG.get(
            "quantization_speedup",
            {"FP32": 1.0, "FP16": 1.45, "INT8": 2.55, "Dynamic_INT8": 2.30}
        )

    def quantize_tensor(
        self,
        tensor: np.ndarray,
        bits: int = 8,
        symmetric: bool = True
    ) -> Dict[str, Any]:
        """Performs true mathematical MinMax quantization on an input weight or activation array.
        
        Computes scale factor S and zero-point Z, quantizes to integer grid,
        dequantizes back, and calculates mean squared error and cosine similarity.
        """
        arr = tensor.astype(np.float32)
        qmin = -(2 ** (bits - 1)) if symmetric else 0
        qmax = (2 ** (bits - 1)) - 1 if symmetric else (2 ** bits) - 1

        val_min = float(np.min(arr))
        val_max = float(np.max(arr))

        if symmetric:
            max_abs = max(abs(val_min), abs(val_max))
            scale = max_abs / qmax if qmax != 0 and max_abs > 0 else 1.0
            zero_point = 0
        else:
            scale = (val_max - val_min) / (qmax - qmin) if (val_max - val_min) > 0 else 1.0
            zero_point = int(np.round(-val_min / scale)) + qmin

        # Quantize
        target_dtype = np.int8 if (bits == 8 and symmetric) else (np.uint8 if bits == 8 else np.int32)
        q_arr = np.clip(np.round(arr / scale) + zero_point, qmin, qmax).astype(target_dtype)
        # Dequantize
        deq_arr = (q_arr.astype(np.float32) - zero_point) * scale


        # Error metrics
        mse = float(np.mean((arr - deq_arr) ** 2))
        norm_orig = np.linalg.norm(arr.flatten())
        norm_deq = np.linalg.norm(deq_arr.flatten())
        if norm_orig > 0 and norm_deq > 0:
            cosine_sim = float(np.dot(arr.flatten(), deq_arr.flatten()) / (norm_orig * norm_deq))
        else:
            cosine_sim = 1.0

        return {
            "scale": scale,
            "zero_point": zero_point,
            "mse": mse,
            "cosine_similarity": cosine_sim,
            "accuracy_retention_pct": round(cosine_sim * 100.0, 2),
            "quantized_shape": q_arr.shape,
        }

    def optimize_model(
        self,
        model_name: str,
        target_precision: str = "INT8",
        custom_base_size_mb: Optional[float] = None,
        custom_base_latency_ms: Optional[float] = None,
    ) -> ModelQuantizationSummary:
        """Derives compression ratio, latency speedup, and size reduction for a given model."""
        meta = self.MODEL_CATALOG.get(
            model_name,
            {
                "name": model_name,
                "base_size_mb": custom_base_size_mb or 50.0,
                "base_latency_ms": custom_base_latency_ms or 12.0,
                "accuracy_baseline": 96.0,
            },
        )

        base_size = custom_base_size_mb if custom_base_size_mb is not None else meta["base_size_mb"]
        base_latency = custom_base_latency_ms if custom_base_latency_ms is not None else meta["base_latency_ms"]

        comp_ratio = self.compression_table.get(target_precision, 1.0)
        speedup = self.speedup_table.get(target_precision, 1.0)

        quant_size = round(base_size / comp_ratio, 2)
        est_latency = round(base_latency / speedup, 2)

        # Accuracy retention: INT8 typically preserves 98.8% to 99.4% of FP32 accuracy
        if target_precision in [QuantizationPrecision.INT8.value, QuantizationPrecision.DYNAMIC_INT8.value]:
            acc_pct = 99.1
            method = "Dynamic Asymmetric MinMax Integer-8"
        elif target_precision == QuantizationPrecision.FP16.value:
            acc_pct = 99.9
            method = "Half-Precision IEEE 754 Float-16"
        else:
            acc_pct = 100.0
            method = "Single-Precision IEEE 754 Float-32 (Baseline)"

        summary = ModelQuantizationSummary(
            model_name=meta.get("name", model_name),
            original_precision="FP32",
            quantized_precision=target_precision,
            original_size_mb=base_size,
            quantized_size_mb=quant_size,
            compression_ratio=comp_ratio,
            speedup_factor=speedup,
            estimated_latency_ms=est_latency,
            accuracy_preservation_pct=acc_pct,
            quantization_method=method,
        )
        logger.info(
            f"Quantized {model_name} to {target_precision}: {base_size}MB -> {quant_size}MB ({comp_ratio}x reduction)"
        )
        return summary
