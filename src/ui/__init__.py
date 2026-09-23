"""Streamlit UI styling, components, charts, and WebRTC streaming modules."""

from src.ui.styles import inject_glassmorphic_styles
from src.ui.charts import (
    render_emotion_radar_chart,
    render_affect_quadrant_chart,
    render_emotion_timeline_chart,
    render_action_units_bar_chart,
)
from src.ui.components import (
    render_header,
    render_metric_card,
    render_affect_summary_badge,
    render_state_indicators,
)
from src.ui.video_processor import MultimodalVideoProcessor

from src.ui.biometric_charts import (
    render_bvp_waveform_chart,
    render_poincare_plot,
    render_autonomic_stress_gauge,
    render_autonomic_balance_bar,
    render_biometric_telemetry_hud_html,
)

__all__ = [
    "inject_glassmorphic_styles",
    "render_emotion_radar_chart",
    "render_affect_quadrant_chart",
    "render_emotion_timeline_chart",
    "render_action_units_bar_chart",
    "render_header",
    "render_metric_card",
    "render_affect_summary_badge",
    "render_state_indicators",
    "MultimodalVideoProcessor",
    "render_bvp_waveform_chart",
    "render_poincare_plot",
    "render_autonomic_stress_gauge",
    "render_autonomic_balance_bar",
    "render_biometric_telemetry_hud_html",
]
