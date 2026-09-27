"""Unit tests for Phase 13 Credibility and Veracity UI Visualizations and HUD."""

import plotly.graph_objects as go

from src.core.credibility_models import (
    MicroLeakageEvent,
    MultimodalPolygraphProfile,
    CredibilitySnapshot,
    FacialVeracityMetrics,
    VoiceStressProfile,
    VeracityTier,
)
from src.ui.credibility_charts import (
    render_credibility_tachometer_gauge,
    render_micro_leakage_timeline,
    render_polygraph_multimodal_stress_radar,
    render_credibility_hud_html,
)


class TestCredibilityUICharts:
    """Tests the Plotly figure constructors and responsive HUD HTML generation."""

    def test_render_credibility_tachometer_gauge(self):
        snapshot = CredibilitySnapshot(
            credibility_score=0.88,
            deception_risk_index=0.12,
            tier=VeracityTier.VERIDICAL_AUTHENTIC.value,
        )
        fig = render_credibility_tachometer_gauge(snapshot)
        assert isinstance(fig, go.Figure)
        assert "CREDIBILITY & VERACITY INDEX" in fig.data[0].title.text

    def test_render_micro_leakage_timeline_empty(self):
        fig = render_micro_leakage_timeline(None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert "MICRO-EXPRESSION LEAKAGE" in fig.layout.title.text

    def test_render_micro_leakage_timeline_with_events(self):
        events = [
            MicroLeakageEvent(timestamp=2.5, duration_ms=110.0, leaked_affect="Fear"),
            MicroLeakageEvent(timestamp=6.8, duration_ms=145.0, leaked_affect="Contempt"),
        ]
        fig = render_micro_leakage_timeline(events)
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].x) == 2

    def test_render_polygraph_multimodal_stress_radar(self):
        polygraph = MultimodalPolygraphProfile(
            facial_incongruence_score=0.35,
            voice_stress_score=0.45,
            pupil_dilation_strain=0.25,
            pacifying_adaptor_score=0.20,
            autonomic_pulse_surge_score=0.30,
        )
        fig = render_polygraph_multimodal_stress_radar(polygraph)
        assert isinstance(fig, go.Figure)
        assert "POLYGRAPHIC" in fig.layout.title.text

    def test_render_credibility_hud_html(self):
        snapshot = CredibilitySnapshot(
            credibility_score=0.82,
            deception_risk_index=0.18,
            tier=VeracityTier.VERIDICAL_AUTHENTIC.value,
            facial_veracity=FacialVeracityMetrics(duchenne_congruence=0.88),
            voice_stress=VoiceStressProfile(stress_index=0.14, cpp_db=13.2),
            active_flags=["Non_Duchenne_Masking"],
        )
        html = render_credibility_hud_html(snapshot)
        assert "FORENSIC VERACITY &amp; CREDIBILITY SENTINEL HUD" in html
        assert "82%" in html
        assert "Non_Duchenne_Masking" in html
