"""Page 7: Edge AI Acceleration, ONNX Runtime Engine, Quantization & Latency Benchmarking Studio."""

import json
import time
import streamlit as st
import numpy as np

from src.core.types import QuantizationPrecision
from src.edge.runtime import ONNXEdgeInferenceEngine
from src.edge.quantizer import ModelQuantizationOptimizer
from src.edge.benchmark import EdgeBenchmarkSuite
from src.ui.styles import inject_modern_styles
from src.ui.components import (
    render_header,
    render_metric_card,
    render_edge_device_card,
    render_benchmark_summary_card,
)
from src.ui.charts import (
    render_edge_latency_waterfall,
    render_quantization_comparison_chart,
    render_edge_throughput_gauge,
)
from src.agent import LLMProviderFactory, LLMProviderConfig

st.set_page_config(
    page_title="Edge Acceleration Studio | EmotionSense",
    page_icon="⚡",
    layout="wide",
)
inject_modern_styles()

render_header(
    "Edge AI Acceleration & Quantization Studio",
    "Sub-10ms On-Device Multimodal Inference, ONNX Runtime Routing & INT8 Dynamic Compression",
)

# Initialize Edge Engine & Suite in session state
if "edge_engine" not in st.session_state:
    st.session_state.edge_engine = ONNXEdgeInferenceEngine()
if "edge_quantizer" not in st.session_state:
    st.session_state.edge_quantizer = ModelQuantizationOptimizer()
if "edge_benchmarks" not in st.session_state:
    st.session_state.edge_benchmarks = []

engine: ONNXEdgeInferenceEngine = st.session_state.edge_engine
quantizer: ModelQuantizationOptimizer = st.session_state.edge_quantizer
bench_suite = EdgeBenchmarkSuite(engine=engine, quantizer=quantizer)

# Top Hardware Diagnostic HUD
device_profile = engine.get_device_profile()
render_edge_device_card(device_profile)

# Sidebar Controls & Runtime Configuration
with st.sidebar:
    st.markdown("### ⚡ Edge Runtime Routing")
    supported_providers = engine.get_supported_providers()

    selected_prov = st.selectbox(
        "Active Execution Provider",
        options=supported_providers,
        index=supported_providers.index(engine.active_provider) if engine.active_provider in supported_providers else 0,
    )
    if selected_prov != engine.active_provider:
        engine.set_provider(selected_prov)
        st.success(f"Switched provider to {selected_prov}")

    selected_prec = st.selectbox(
        "Target Quantization Precision",
        options=[
            QuantizationPrecision.INT8.value,
            QuantizationPrecision.DYNAMIC_INT8.value,
            QuantizationPrecision.FP16.value,
            QuantizationPrecision.FP32.value,
        ],
        index=0,
    )

    benchmark_iterations = st.slider("Benchmark Iterations", min_value=10, max_value=100, value=30, step=10)

    st.markdown("---")
    st.markdown("### 🎯 Edge Latency SLAs")
    st.markdown("""
    - **Acoustic SER**: `< 8.0 ms`
    - **Dialogue NLP**: `< 12.0 ms`
    - **Vision Mesh**: `< 15.0 ms`
    - **Cross-Modal (CMAF)**: `< 20.0 ms`
    - **Target FPS**: `≥ 30.0 FPS`
    """)

# Main Studio Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Latency Benchmark & Stress Test",
    "🗜️ Model Quantization & Compression",
    "📦 Edge Manifest & Export",
    "🤖 Edge SLM & Agent Profiling",
])

with tab1:
    col_ctrl, col_act = st.columns([3, 1])
    with col_ctrl:
        st.markdown("#### ⏱️ Real-Time Pipeline Latency Telemetry")
        st.caption("Measure P50 (median), P95, and P99 tail latency percentiles across all 4 multimodal models under simulated target hardware load.")

    with col_act:
        run_bench = st.button("⚡ Run Full Benchmark Suite", use_container_width=True, type="primary")

    if run_bench or not st.session_state.edge_benchmarks:
        with st.spinner("Executing stress test iterations & telemetry sampling..."):
            results = bench_suite.run_all_models(
                provider=engine.active_provider,
                precision=selected_prec,
                iterations=benchmark_iterations,
            )
            st.session_state.edge_benchmarks = results

    # Metrics Summary Cards
    bm_list = st.session_state.edge_benchmarks
    if bm_list:
        avg_lat = float(np.mean([b.mean_latency_ms for b in bm_list]))
        overall_fps = float(np.mean([b.fps_throughput for b in bm_list]))
        sla_pass_count = sum(1 for b in bm_list if b.sla_compliant)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Mean Latency", f"{avg_lat:.1f} ms", delta="Sub-15ms Target", color="#3b82f6")
        with m2:
            render_metric_card("Avg Throughput", f"{overall_fps:.1f} FPS", delta="Real-Time SLA", color="#10b981")
        with m3:
            render_metric_card("SLA Compliance", f"{sla_pass_count} / {len(bm_list)} Passed", delta="Zero Dropped Frames", color="#0ea5e9")
        with m4:
            render_metric_card("Precision Mode", selected_prec, delta=engine.active_provider.split("Execution")[0], color="#f59e0b")

        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

        chart_c1, chart_c2 = st.columns([2.5, 1.2])
        with chart_c1:
            fig_lat = render_edge_latency_waterfall(bm_list)
            st.plotly_chart(fig_lat, use_container_width=True)
        with chart_c2:
            fig_fps = render_edge_throughput_gauge(overall_fps, target_fps=60.0)
            st.plotly_chart(fig_fps, use_container_width=True)

        st.markdown("#### 📋 Detailed Model Benchmark Telemetry")
        for b in bm_list:
            render_benchmark_summary_card(b)

with tab2:
    st.markdown("#### 🗜️ Dynamic Post-Training Quantization (PTQ)")
    st.caption("Evaluate memory footprint compression ratios, inference speedups, and accuracy retention across FP32, FP16, and INT8 formats.")

    models_to_quantize = ["vision_mesh", "audio_ser", "text_nlp", "cross_modal_cmaf"]
    summaries = [quantizer.optimize_model(m, target_precision=selected_prec) for m in models_to_quantize]

    fig_quant = render_quantization_comparison_chart(summaries)
    st.plotly_chart(fig_quant, use_container_width=True)

    # Real Tensor Quantization Sandbox
    st.markdown("---")
    st.markdown("#### 🔬 Mathematical MinMax Quantization Sandbox")
    st.caption("Observe real-time array quantization, dynamic scale factor derivation, zero-point alignment, and cosine similarity retention.")

    sb_c1, sb_c2 = st.columns([2, 1])
    with sb_c1:
        test_tensor_type = st.radio(
            "Activation / Weight Distribution",
            ["Gaussian Normal (μ=0, σ=1)", "Uniform Random [-2.5, 2.5]", "Sparse Attention Weights [0, 1]"],
            horizontal=True,
        )
    with sb_c2:
        quant_bits = st.selectbox("Quantization Bit Depth", [8, 4], index=0)

    if "Gaussian" in test_tensor_type:
        raw_tensor = np.random.randn(256).astype(np.float32)
    elif "Uniform" in test_tensor_type:
        raw_tensor = np.random.uniform(-2.5, 2.5, 256).astype(np.float32)
    else:
        raw_tensor = np.random.uniform(0.0, 1.0, 256).astype(np.float32)

    q_eval = quantizer.quantize_tensor(raw_tensor, bits=quant_bits, symmetric=True)

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        render_metric_card("Scale Factor (S)", f"{q_eval['scale']:.6f}", color="#3b82f6")
    with q2:
        render_metric_card("Zero Point (Z)", f"{q_eval['zero_point']}", color="#0ea5e9")
    with q3:
        render_metric_card("Reconstruction MSE", f"{q_eval['mse']:.6f}", color="#f59e0b")
    with q4:
        render_metric_card("Cosine Similarity", f"{q_eval['accuracy_retention_pct']:.2f}%", delta="Preserved Signal", color="#10b981")

with tab3:
    st.markdown("#### 📦 Export Edge Deployment Manifest")
    st.caption("Generate a standard JSON deployment manifest bundle specifying hardware profile, model quantization specs, and verified SLA benchmarks.")

    target_env = st.selectbox(
        "Target Edge Deployment Architecture",
        [
            "Edge/Mobile (Android NNAPI / iOS CoreML)",
            "Browser WebAssembly (Wasm + SIMD)",
            "Embedded Linux / NVIDIA Jetson (DirectML / TensorRT)",
            "Enterprise On-Premises Server (CPU Threaded)",
        ],
    )

    if st.button("🔨 Generate Deployment Manifest Bundle", type="primary"):
        manifest = bench_suite.generate_manifest(target_env=target_env, target_precision=selected_prec)
        manifest_json = json.dumps(manifest.to_dict(), indent=2)

        st.success(f"Deployment manifest `{manifest.manifest_id}` generated successfully!")
        st.download_button(
            label="⬇️ Download edge_manifest.json",
            data=manifest_json,
            file_name=f"emotionsense_edge_manifest_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.code(manifest_json, language="json")

with tab4:
    st.markdown("#### 🤖 Local Edge SLM & Clinical Reasoner Profiling")
    st.caption("Benchmark on-device Small Language Models (SLM) for zero-cloud data sovereignty, latency, and tokens-per-second throughput.")

    slm_col1, slm_col2 = st.columns([1.5, 3.5])
    with slm_col1:
        target_slm = st.selectbox(
            "Target Local Engine",
            ["rule_based", "ollama"],
            format_func=lambda x: "🛡️ Rule-Based Expert (0ms / Deterministic)" if x == "rule_based" else "🦙 Ollama Local Daemon (LLaMA 3.2 / Mistral)",
            key="edge_slm_sel"
        )
        test_slm_btn = st.button("⚡ Profile SLM Generation", use_container_width=True, type="primary")

    if test_slm_btn:
        t0 = time.perf_counter()
        prov = LLMProviderFactory.get_provider(LLMProviderConfig(provider_name=target_slm))
        res_text = prov.generate("Evaluate clinical risk for session with minor valence dips")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        with slm_col2:
            st.markdown(f"**Engine Active:** `{prov.provider_name}` (`{prov.model_name}`)")
            slm_m1, slm_m2, slm_m3 = st.columns(3)
            with slm_m1:
                render_metric_card("Inference Latency", f"{dt_ms:.1f} ms", delta="Sub-10ms Compliant" if dt_ms < 10 else "Local SLM")
            with slm_m2:
                tok_estimate = len(res_text.split()) * 1.3
                tps = (tok_estimate / (dt_ms / 1000.0)) if dt_ms > 0 else 999.0
                render_metric_card("Throughput", f"{min(tps, 500.0):.1f} tok/s", delta="Generation Speed")
            with slm_m3:
                render_metric_card("Data Sovereignty", "100% On-Device", delta="Zero Cloud Exfiltration")

            st.markdown(f"""
            <div class="es-panel" style="margin-top: 10px; font-size: 0.85rem; border-left: 3px solid #10b981;">
                <b>Raw Output Payload Preview:</b><br/>
                <code style="color: #cbd5e1; font-size: 0.78rem;">{res_text[:280]}...</code>
            </div>
            """, unsafe_allow_html=True)
