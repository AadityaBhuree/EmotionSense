"""Unit tests for Phase 11 UI Cognitive Visualizations and Charts."""

import plotly.graph_objects as go

from src.core.cognitive_models import (
    WorkloadTier,
    PupillometryMetrics,
    BlinkDynamics,
    GazeTelemetry,
    NASATLXDimensions,
    CognitiveWorkloadRecord,
    OculomotorSnapshot,
)
from src.ui.cognitive_charts import (
    render_gaze_dispersion_chart,
    render_nasa_tlx_radar,
    render_cognitive_workload_gauge,
    render_oculomotor_hud_html,
)


def test_render_gaze_dispersion_chart_with_data():
    pts = [(0.48, 0.51), (0.50, 0.49), (0.52, 0.53), (0.49, 0.50)]
    gaze = GazeTelemetry(screen_x=0.51, screen_y=0.50, dispersion_area=0.08)
    fig = render_gaze_dispersion_chart(pts, current_gaze=gaze)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2
    assert "OCULOMOTOR GAZE DISPERSION" in fig.layout.title.text


def test_render_gaze_dispersion_chart_empty():
    fig = render_gaze_dispersion_chart([])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_render_nasa_tlx_radar():
    dims = NASATLXDimensions(mental_demand=65.0, temporal_demand=55.0, effort=70.0, frustration=30.0)
    fig = render_nasa_tlx_radar(dims)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatterpolar"
    assert "NASA-TLX WORKLOAD RADAR" in fig.layout.title.text


def test_render_cognitive_workload_gauge():
    record = CognitiveWorkloadRecord(workload_index=0.68, tier=WorkloadTier.HIGH_EFFORT.value)
    fig = render_cognitive_workload_gauge(record)

    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "indicator"
    assert fig.data[0].value == 68.0


def test_render_oculomotor_hud_html():
    snapshot = OculomotorSnapshot(
        pupillometry=PupillometryMetrics(pupil_diameter_ratio=0.46, dilation_change_pct=15.0),
        blinks=BlinkDynamics(blink_rate_bpm=14.0, perclos=0.08, micro_sleep_detected=False),
        gaze=GazeTelemetry(fixation_duration_ms=450.0, saccade_velocity_deg_s=140.0),
        workload=CognitiveWorkloadRecord(workload_index=0.45, tier=WorkloadTier.OPTIMAL_ENGAGEMENT.value),
    )
    html = render_oculomotor_hud_html(snapshot)

    assert isinstance(html, str)
    assert "0.460" in html
    assert "14.0 BPM" in html
    assert "OPTIMAL ENGAGEMENT" in html
    assert "OCULOMOTOR & COGNITIVE WORKLOAD HUD" in html
