"""Edge AI Acceleration and Quantization Router."""

from fastapi import APIRouter
from src.api.deps import edge_engine, edge_quantizer, edge_bench_suite
from src.api.v1.schemas import EdgeBenchmarkRequest, EdgeQuantizeRequest

router = APIRouter(tags=["Edge Acceleration"])


@router.get("/api/edge/hardware")
async def get_edge_hardware():
    """Returns detected hardware specifications, CPU threads, RAM, and active ONNX execution providers."""
    profile = edge_engine.get_device_profile()
    return {"status": "success", "profile": profile.to_dict()}


@router.post("/api/edge/benchmark")
async def benchmark_edge_model(req: EdgeBenchmarkRequest):
    """Executes latency stress testing across iterations, calculating P50/P95/P99 percentiles and FPS throughput."""
    res = edge_bench_suite.run_benchmark(
        model_name=req.model_name or "vision_mesh",
        provider=req.provider,
        precision=req.precision or "INT8",
        iterations=req.iterations or 20,
    )
    return {"status": "success", "benchmark": res.to_dict()}


@router.post("/api/edge/quantize")
async def quantize_edge_model(req: EdgeQuantizeRequest):
    """Computes dynamic post-training quantization metrics, size reduction ratios, and accuracy retention."""
    summary = edge_quantizer.optimize_model(
        model_name=req.model_name,
        target_precision=req.target_precision or "INT8",
    )
    return {"status": "success", "quantization": summary.to_dict()}
