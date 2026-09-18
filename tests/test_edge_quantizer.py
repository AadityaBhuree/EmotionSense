"""Unit tests for Model Quantization Optimizer, MinMax scaling, and compression ratios."""

import numpy as np

from src.edge.quantizer import ModelQuantizationOptimizer


def test_quantize_tensor_symmetric():
    opt = ModelQuantizationOptimizer()
    tensor = np.random.uniform(-3.0, 3.0, 500).astype(np.float32)

    result = opt.quantize_tensor(tensor, bits=8, symmetric=True)

    assert result["scale"] > 0.0
    assert result["zero_point"] == 0
    assert result["mse"] >= 0.0
    assert result["cosine_similarity"] > 0.95
    assert result["accuracy_retention_pct"] > 95.0


def test_quantize_tensor_asymmetric():
    opt = ModelQuantizationOptimizer()
    tensor = np.random.uniform(2.0, 7.0, 500).astype(np.float32)

    result = opt.quantize_tensor(tensor, bits=8, symmetric=False)

    assert result["scale"] > 0.0
    assert isinstance(result["zero_point"], int)
    assert result["mse"] >= 0.0
    assert result["cosine_similarity"] > 0.95



def test_model_quantization_optimization_catalog():
    opt = ModelQuantizationOptimizer()

    summary_int8 = opt.optimize_model("vision_mesh", target_precision="INT8")
    assert summary_int8.quantized_size_mb < summary_int8.original_size_mb
    assert summary_int8.compression_ratio > 3.0
    assert summary_int8.speedup_factor > 2.0
    assert summary_int8.accuracy_preservation_pct >= 98.0

    summary_fp16 = opt.optimize_model("text_nlp", target_precision="FP16")
    assert summary_fp16.compression_ratio == 2.0
    assert summary_fp16.accuracy_preservation_pct > 99.0
