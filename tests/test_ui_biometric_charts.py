"""Unit tests for Phase 10 Biometric Visualizations and Charts."""

import plotly.graph_objects as go

from src.core.biometric_models import (
    PulseMeasurement,
    HRVMetrics,
    RespirationMetrics,
    AutonomicStressRecord,
    BiometricTelemetry,
)
from src.ui.biometric_charts import (
    render_bvp_waveform_chart,
    render_poincare_plot,
    render_autonomic_stress_gauge,
    render_autonomic_balance_bar,
    render_biometric_telemetry_hud_html,
)


def test_render_bvp_waveform_chart_with_data():
    sig = [0.1, 0.3, 0.8, 0.4, -0.2, -0.5, 0.1, 0.7, 0.2]
    pulse = PulseMeasurement(bpm=72.0, signal_quality_snr=12.4)
    fig = render_bvp_waveform_chart(sig, pulse=pulse)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert "OPTICAL PHOTOPLETHYSMOGRAM" in fig.layout.title.text


def test_render_bvp_waveform_chart_empty():
    fig = render_bvp_waveform_chart([])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_render_poincare_plot():
    rr = [800.0, 810.0, 790.0, 820.0, 805.0, 795.0, 815.0]
    fig = render_poincare_plot(rr)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # Identity line + scatter markers


def test_render_poincare_plot_empty():
    fig = render_poincare_plot([])
    assert isinstance(fig, go.Figure)


def test_render_autonomic_stress_gauge():
    stress = AutonomicStressRecord(stress_index=0.45, classification="Optimal_Alertness")
    fig = render_autonomic_stress_gauge(stress)

    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "indicator"
    assert fig.data[0].value == 45.0


def test_render_autonomic_balance_bar():
    stress = AutonomicStressRecord(sympathetic_tone=0.40, parasympathetic_tone=0.60)
    fig = render_autonomic_balance_bar(stress)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Sympathetic + Parasympathetic bars


def test_render_biometric_telemetry_hud_html():
    telem = BiometricTelemetry(
        pulse=PulseMeasurement(bpm=75.2, signal_quality_snr=11.8),
        hrv=HRVMetrics(rmssd_ms=38.4, baevsky_stress_index=95.0),
        respiration=RespirationMetrics(rpm=15.2),
        autonomic_stress=AutonomicStressRecord(stress_index=0.32, classification="Optimal_Alertness"),
    )
    html = render_biometric_telemetry_hud_html(telem)

    assert isinstance(html, str)
    assert "75.2" in html
    assert "38.4 ms" in html
    assert "OPTIMAL_ALERTNESS" in html
    assert "AUTONOMIC BIOMETRICS" in html
