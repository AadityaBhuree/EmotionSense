"""Tests for DyadicInteractionAnalyzer interpersonal synchrony and rapport engine."""

from src.fusion.interaction_dynamics import DyadicInteractionAnalyzer
from src.core.types import DiarizationResult, DyadicInteractionMetrics, SpeakerTurn


def test_analyzer_initialization():
    analyzer = DyadicInteractionAnalyzer()
    assert analyzer.synchrony_window_sec == 30.0
    assert analyzer.mimicry_lag_window_sec == 2.0
    assert analyzer.sample_rate_hz == 2.0


def test_insufficient_samples_handling():
    analyzer = DyadicInteractionAnalyzer()
    res1 = analyzer.analyze([], [])
    assert isinstance(res1, DyadicInteractionMetrics)
    assert res1.rapport_score == 50.0
    assert res1.resonance_category == "Insufficient Data"

    # Single sample
    res2 = analyzer.analyze([{"time": 0.0, "valence": 0.5}], [{"time": 0.0, "valence": 0.5}])
    assert res2.resonance_category == "Insufficient Data"

    # Non-overlapping timelines (0-2s vs 10-12s)
    samples_a = [{"time": 0.0, "valence": 0.5}, {"time": 2.0, "valence": 0.5}]
    samples_b = [{"time": 10.0, "valence": 0.5}, {"time": 12.0, "valence": 0.5}]
    res3 = analyzer.analyze(samples_a, samples_b)
    assert res3.resonance_category == "Insufficient Data"


def test_high_collaborative_rapport():
    analyzer = DyadicInteractionAnalyzer()

    # Two participants with mutually positive, aligned valence and mutual smiles
    times = [float(i) for i in range(15)]
    samples_a = [
        {"timestamp": t, "valence": 0.6 + 0.1 * (t % 3), "arousal": 0.5, "smile": 0.8, "yaw": 2.0}
        for t in times
    ]
    samples_b = [
        {"timestamp": t, "valence": 0.65 + 0.1 * (t % 3), "arousal": 0.48, "smile": 0.75, "yaw": -3.0}
        for t in times
    ]

    diarization = DiarizationResult(
        speakers=["Speaker_0", "Speaker_1"],
        speaker_durations={"Speaker_0": 7.0, "Speaker_1": 7.5},
        dominance_ratios={"Speaker_0": 0.48, "Speaker_1": 0.52},
        interruption_count=0,
    )

    res = analyzer.analyze(samples_a, samples_b, diarization)
    assert res.rapport_score >= 70.0
    assert res.valence_synchrony > 0.7
    assert res.conversational_balance > 0.85
    assert res.dominance_speaker == "balanced"
    assert "High Collaborative Resonance" in res.resonance_category or "Empathic" in res.resonance_category


def test_discordant_and_dissonant_exchange():
    analyzer = DyadicInteractionAnalyzer()

    # Participant A happy, Participant B increasingly negative (opposite valence trajectory)
    times = [float(i) for i in range(15)]
    samples_a = [
        {"timestamp": t, "valence": 0.2 + (t * 0.04), "arousal": 0.4, "smile": 0.5, "yaw": 0.0}
        for t in times
    ]
    samples_b = [
        {"timestamp": t, "valence": 0.2 - (t * 0.05), "arousal": 0.6, "smile": 0.0, "yaw": 20.0}
        for t in times
    ]

    # High interruption count
    diarization = DiarizationResult(
        speakers=["Speaker_0", "Speaker_1"],
        speaker_durations={"Speaker_0": 10.0, "Speaker_1": 2.0},
        interruption_count=6,
    )

    res = analyzer.analyze(samples_a, samples_b, diarization)
    assert res.valence_synchrony < 0.0  # Inverted trajectory
    assert res.conversational_balance < 0.5  # Skewed
    assert res.rapport_score < 55.0
    assert len(res.summary_notes) > 0


def test_lagged_smile_mimicry():
    analyzer = DyadicInteractionAnalyzer(sample_rate_hz=2.0)

    # Participant A smiles at t=2.0s to t=5.0s
    # Participant B mimics and smiles at t=3.0s to t=6.0s (1.0s reaction lag)
    times = [float(i) * 0.5 for i in range(30)]  # 15 seconds at 0.5s intervals
    samples_a = []
    samples_b = []

    for t in times:
        smile_a = 0.9 if 2.0 <= t <= 5.0 else 0.05
        smile_b = 0.85 if 3.0 <= t <= 6.0 else 0.05
        samples_a.append({"timestamp": t, "valence": 0.4, "arousal": 0.3, "smile": smile_a, "yaw": 0.0})
        samples_b.append({"timestamp": t, "valence": 0.4, "arousal": 0.3, "smile": smile_b, "yaw": 0.0})

    res = analyzer.analyze(samples_a, samples_b)
    # The lagged cross-correlation should identify the smile mimicry
    assert res.mimicry_index > 0.35


def test_transition_latency_from_speaker_turns():
    analyzer = DyadicInteractionAnalyzer()
    samples = [{"timestamp": float(i), "valence": 0.2, "arousal": 0.2} for i in range(10)]

    turns = [
        SpeakerTurn(speaker_id="Speaker_0", start_time=0.0, end_time=2.0, duration=2.0),
        SpeakerTurn(speaker_id="Speaker_1", start_time=2.4, end_time=5.0, duration=2.6),
        SpeakerTurn(speaker_id="Speaker_0", start_time=5.8, end_time=8.0, duration=2.2),
    ]
    diarization = DiarizationResult(
        turns=turns,
        speakers=["Speaker_0", "Speaker_1"],
        speaker_durations={"Speaker_0": 4.2, "Speaker_1": 2.6},
        interruption_count=0,
    )

    res = analyzer.analyze(samples, samples, diarization)
    # Latencies: 2.4 - 2.0 = 0.4s, 5.8 - 5.0 = 0.8s -> avg = 0.6s
    assert abs(res.turn_transition_latency - 0.6) < 0.05
