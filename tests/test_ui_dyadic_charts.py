"""Tests for Phase 5 Dyadic UI Charts and Component Renderers."""

import plotly.graph_objects as go
from src.ui.charts import (
    render_dyadic_synchrony_chart,
    render_conversational_dominance_pie,
    render_rapport_gauge,
    render_turn_taking_timeline,
)
from src.core.types import SpeakerTurn


def test_dyadic_synchrony_chart():
    times = [0.0, 1.0, 2.0, 3.0]
    val_a = [0.2, 0.4, 0.6, 0.5]
    val_b = [0.1, 0.3, 0.5, 0.6]

    fig = render_dyadic_synchrony_chart(times, val_a, val_b, "Candidate", "Interviewer", 0.85)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].name == "Candidate"
    assert fig.data[1].name == "Interviewer"


def test_conversational_dominance_pie():
    durs = {"Speaker_0": 45.0, "Speaker_1": 55.0}
    fig = render_conversational_dominance_pie(durs)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].hole == 0.55


def test_rapport_gauge():
    fig = render_rapport_gauge(82.5, "High Collaborative Resonance")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].value == 82.5


def test_turn_taking_timeline():
    turns = [
        SpeakerTurn(speaker_id="Speaker_0", start_time=0.0, end_time=2.0, duration=2.0),
        SpeakerTurn(speaker_id="Speaker_1", start_time=2.2, end_time=4.5, duration=2.3),
    ]
    fig = render_turn_taking_timeline(turns)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
