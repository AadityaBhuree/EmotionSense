"""Unit tests for Phase 8 Edge telemetry charts and gauge visualizations."""

import plotly.graph_objects as go

from src.core.types import InferenceBenchmarkResult, ModelQuantizationSummary
from src.ui.charts import (
    render_edge_latency_waterfall,
    render_quantization_comparison_chart,
    render_edge_throughput_gauge,
)


def test_render_edge_latency_waterfall():
    benchmarks = [
        InferenceBenchmarkResult(
            model_name="vision_mesh",
            p50_latency_ms=12.5,
            p95_latency_ms=14.8,
            p99_latency_ms=16.2,
        ),
        InferenceBenchmarkResult(
            model_name="audio_ser",
            p50_latency_ms=6.1,
            p95_latency_ms=7.8,
            p99_latency_ms=8.5,
        ),
    ]

    fig = render_edge_latency_waterfall(benchmarks)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # P50, P95, P99 traces


def test_render_quantization_comparison_chart():
    summaries = [
        ModelQuantizationSummary(
            model_name="Face Mesh AU",
            original_size_mb=42.5,
            quantized_size_mb=10.9,
            compression_ratio=3.9,
            speedup_factor=2.5,
        ),
        ModelQuantizationSummary(
            model_name="Wav2Vec2 SER",
            original_size_mb=95.0,
            quantized_size_mb=24.3,
            compression_ratio=3.9,
            speedup_factor=2.5,
        ),
    ]

    fig = render_quantization_comparison_chart(summaries)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Baseline FP32 & Quantized INT8 traces


def test_render_edge_throughput_gauge():
    fig = render_edge_throughput_gauge(fps=45.2, target_fps=60.0)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "indicator"
    assert fig.data[0].value == 45.2
