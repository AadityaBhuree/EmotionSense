"""Unit tests for ONNX Edge Inference Engine, provider routing, and warm-up routines."""

import numpy as np

from src.core.types import EdgeExecutionProvider
from src.edge.runtime import ONNXEdgeInferenceEngine


def test_edge_engine_initialization_and_profile():
    engine = ONNXEdgeInferenceEngine()
    profile = engine.get_device_profile()

    assert profile.device_name is not None
    assert profile.provider in engine.get_supported_providers()
    assert profile.available_threads > 0
    assert profile.total_ram_mb > 0
    assert EdgeExecutionProvider.CPU.value in profile.supported_providers


def test_edge_engine_provider_switching():
    engine = ONNXEdgeInferenceEngine()
    assert engine.set_provider(EdgeExecutionProvider.CPU.value) is True
    assert engine.active_provider == EdgeExecutionProvider.CPU.value

    # DirectML / CUDA / WASM valid provider switching
    assert engine.set_provider(EdgeExecutionProvider.DIRECTML.value) is True
    assert engine.active_provider == EdgeExecutionProvider.DIRECTML.value

    # Invalid provider returns False
    assert engine.set_provider("NonExistentProvider") is False
    assert engine.active_provider == EdgeExecutionProvider.DIRECTML.value


def test_edge_engine_warmup():
    engine = ONNXEdgeInferenceEngine()
    warmup_ms = engine.warm_up("vision_mesh")
    assert warmup_ms > 0.0


def test_edge_engine_synthetic_inference():
    engine = ONNXEdgeInferenceEngine()
    engine.set_provider(EdgeExecutionProvider.CPU.value)

    dummy_input = np.random.randn(1, 64).astype(np.float32)
    res_fp32 = engine.run_synthetic_inference("audio_ser", input_data=dummy_input, precision="FP32")

    assert res_fp32["status"] == "SUCCESS"
    assert res_fp32["model_name"] == "audio_ser"
    assert res_fp32["precision"] == "FP32"
    assert res_fp32["latency_ms"] > 0.0

    # INT8 should be faster on average than FP32
    res_int8 = engine.run_synthetic_inference("audio_ser", input_data=dummy_input, precision="INT8")
    assert res_int8["latency_ms"] > 0.0


def test_edge_benchmark_suite_rppg():
    from src.edge.benchmark import EdgeBenchmarkSuite
    suite = EdgeBenchmarkSuite()
    res = suite.evaluate_rppg_pipeline(iterations=5, window_frames=60)

    assert "mean_latency_ms" in res
    assert "p50_latency_ms" in res
    assert "stage_latencies_ms" in res
    assert res["zero_cloud_certified"] is True
    assert res["mean_latency_ms"] > 0.0


def test_edge_benchmark_suite_oculomotor():
    from src.edge.benchmark import EdgeBenchmarkSuite
    suite = EdgeBenchmarkSuite()
    res = suite.evaluate_oculomotor_pipeline(iterations=10)

    assert "mean_latency_ms" in res
    assert "p50_latency_ms" in res
    assert "stage_latencies_ms" in res
    assert "iris_pupillometry_calc" in res["stage_latencies_ms"]
    assert res["zero_cloud_certified"] is True
    assert res["sla_target_ms"] == 4.0
    assert res["mean_latency_ms"] > 0.0


def test_edge_benchmark_suite_somatosensory():
    from src.edge.benchmark import EdgeBenchmarkSuite
    suite = EdgeBenchmarkSuite()
    res = suite.evaluate_somatosensory_pipeline(iterations=10)

    assert "mean_latency_ms" in res
    assert "p50_latency_ms" in res
    assert "stage_latencies_ms" in res
    assert "posture_angle_kinematics" in res["stage_latencies_ms"]
    assert "microgesture_adaptor_heuristic" in res["stage_latencies_ms"]
    assert "kinetic_fidgeting_flux" in res["stage_latencies_ms"]
    assert "pai_multimodal_fusion" in res["stage_latencies_ms"]
    assert res["zero_cloud_certified"] is True
    assert res["sla_target_ms"] == 3.5
    assert res["mean_latency_ms"] > 0.0



