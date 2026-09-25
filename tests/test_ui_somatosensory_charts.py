"""Unit tests for Phase 12 Somatosensory UI Visualizations and HUD."""

import pytest
import plotly.graph_objects as go

from src.core.somatosensory_models import (
    PostureState,
    AdaptorType,
    AdaptorCategory,
    PsychomotorTier,
    PosturalMetrics,
    MicroGestureAdaptor,
    FidgetingDynamics,
    KinesicExpressivity,
    PsychomotorAgitationIndex,
    SomatosensorySnapshot,
)
from src.ui.somatosensory_charts import (
    render_postural_ergonomics_diagram,
    render_adaptor_timeline_chart,
    render_fidgeting_waveform,
    render_psychomotor_agitation_gauge,
    render_somatosensory_hud_html,
)


class TestSomatosensoryUICharts:
    """Tests Plotly figure generation and HTML formatting for somatosensory telemetry."""

    def test_render_postural_ergonomics_diagram(self):
        pm = PosturalMetrics(
            forward_head_angle_deg=52.0,
            spinal_tilt_deg=3.5,
            shoulder_elevation_asymmetry=0.04,
            slump_index=0.20,
            posture_state=PostureState.UPRIGHT.value,
        )
        fig = render_postural_ergonomics_diagram(pm)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2
        assert "POSTURAL ERGONOMICS" in fig.layout.title.text

    def test_render_adaptor_timeline_chart_empty_and_populated(self):
        # Empty fallback
        fig_empty = render_adaptor_timeline_chart([])
        assert isinstance(fig_empty, go.Figure)
        assert len(fig_empty.data) == 1

        # Populated with adaptors
        adaptors = [
            MicroGestureAdaptor(
                adaptor_type=AdaptorType.CHIN_SUPPORT.value,
                category=AdaptorCategory.EVALUATIVE_COGNITIVE.value,
                proximity_distance=0.12,
                active=True,
            ),
            MicroGestureAdaptor(
                adaptor_type=AdaptorType.NECK_TOUCH.value,
                category=AdaptorCategory.PACIFYING_STRESS.value,
                proximity_distance=0.18,
                active=True,
            ),
        ]
        fig_pop = render_adaptor_timeline_chart(adaptors)
        assert isinstance(fig_pop, go.Figure)
        assert "Chin Support" in fig_pop.data[0].y

    def test_render_fidgeting_waveform(self):
        # Fallback empty
        fig_empty = render_fidgeting_waveform()
        assert isinstance(fig_empty, go.Figure)

        # Populated history
        history = [(float(i), 0.01 + 0.002 * (i % 3)) for i in range(20)]
        fd = FidgetingDynamics(restlessness_score=0.45, is_fidgeting=True)
        fig_pop = render_fidgeting_waveform(history, current_fidgeting=fd)
        assert isinstance(fig_pop, go.Figure)
        assert len(fig_pop.data[0].x) == 20

    def test_render_psychomotor_agitation_gauge_tiers(self):
        for tier, idx in [
            (PsychomotorTier.COMPOSED.value, 0.15),
            (PsychomotorTier.RESTLESS_MILD.value, 0.35),
            (PsychomotorTier.AGITATED_HIGH.value, 0.65),
            (PsychomotorTier.ACUTE_MOTOR_STORM.value, 0.88),
        ]:
            pai = PsychomotorAgitationIndex(agitation_index=idx, tier=tier)
            fig = render_psychomotor_agitation_gauge(pai)
            assert isinstance(fig, go.Figure)
            assert fig.data[0].value == pytest.approx(idx * 100.0)

    def test_render_somatosensory_hud_html(self):
        snap = SomatosensorySnapshot(
            posture=PosturalMetrics(posture_state=PostureState.UPRIGHT.value, slump_index=0.18),
            primary_adaptor=MicroGestureAdaptor(
                adaptor_type=AdaptorType.CHIN_SUPPORT.value,
                category=AdaptorCategory.EVALUATIVE_COGNITIVE.value,
                active=True,
            ),
            fidgeting=FidgetingDynamics(restlessness_score=0.22, is_fidgeting=False),
            expressivity=KinesicExpressivity(expressivity_score=0.55),
            agitation=PsychomotorAgitationIndex(
                agitation_index=0.20,
                tier=PsychomotorTier.COMPOSED.value,
                contributing_factors=["Stable Composure", "Ergonomic Alignment"],
            ),
        )
        html = render_somatosensory_hud_html(snap)
        assert "SOMATOSENSORY & KINESICS TELEMETRY HUD" in html
        assert "UPRIGHT" in html
        assert "CHIN SUPPORT" in html
        assert "COMPOSED" in html
        assert "Stable Composure" in html
