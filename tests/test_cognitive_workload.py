"""Unit tests for Phase 11 Oculomotor Telemetry, Pupillometry & Cognitive Workload Engine."""

import pytest
import numpy as np

from src.core.cognitive_models import (
    WorkloadTier,
    PupillometryMetrics,
    BlinkDynamics,
    GazeTelemetry,
    OculomotorSnapshot,
)
from src.analytics.oculometrics import OculomotorEngine


@pytest.fixture
def oculomotor_engine():
    return OculomotorEngine(ear_closed_threshold=0.20, baseline_pir=0.40)


@pytest.fixture
def synthetic_open_eye():
    # 6 points [p0, p1, p2, p3, p4, p5] with wide vertical separation (open)
    return np.array([
        [0.0, 0.0],    # p0 outer
        [0.3, 0.25],   # p1 top-outer
        [0.7, 0.25],   # p2 top-inner
        [1.0, 0.0],    # p3 inner
        [0.7, -0.25],  # p4 bottom-inner
        [0.3, -0.25],  # p5 bottom-outer
    ])


@pytest.fixture
def synthetic_closed_eye():
    # Narrow vertical separation (closed)
    return np.array([
        [0.0, 0.0],
        [0.3, 0.04],
        [0.7, 0.04],
        [1.0, 0.0],
        [0.7, -0.04],
        [0.3, -0.04],
    ])


def test_calculate_ear(synthetic_open_eye, synthetic_closed_eye):
    ear_open = OculomotorEngine.calculate_ear(synthetic_open_eye)
    ear_closed = OculomotorEngine.calculate_ear(synthetic_closed_eye)

    assert ear_open > 0.25
    assert ear_closed < 0.15
    assert ear_open > ear_closed


def test_evaluate_blinks_normal_and_microsleep(oculomotor_engine):
    t0 = 1000.0
    # Frame 1: Eyes open
    b1 = oculomotor_engine.evaluate_blinks(0.32, 0.31, timestamp=t0)
    assert not b1.micro_sleep_detected

    # Eyes close at t0 + 0.1
    b2 = oculomotor_engine.evaluate_blinks(0.12, 0.11, timestamp=t0 + 0.1)
    assert not b2.micro_sleep_detected

    # Eyes open after 150ms -> normal blink
    b3 = oculomotor_engine.evaluate_blinks(0.30, 0.29, timestamp=t0 + 0.25)
    assert not b3.micro_sleep_detected
    assert len(oculomotor_engine.logged_blinks) == 1

    # Eyes close and remain closed for 600ms -> micro-sleep alert!
    oculomotor_engine.evaluate_blinks(0.10, 0.10, timestamp=t0 + 1.0)
    b_micro = oculomotor_engine.evaluate_blinks(0.09, 0.10, timestamp=t0 + 1.65)
    assert b_micro.micro_sleep_detected is True


def test_evaluate_blinks_perclos(oculomotor_engine):
    t = 1000.0
    # Simulate series with high closure
    for i in range(20):
        ear = 0.10 if i % 2 == 0 else 0.30
        res = oculomotor_engine.evaluate_blinks(ear, ear, timestamp=t + (i * 0.1))
    assert res.perclos > 0.30


def test_evaluate_pupillometry(oculomotor_engine):
    # Baseline pupil
    p_base = oculomotor_engine.evaluate_pupillometry(0.40)
    assert p_base.dilation_change_pct == 0.0
    assert p_base.cognitive_pupillary_response == 0.0

    # Dilated pupil under mental effort (PIR = 0.52 -> +30% dilation)
    p_dilated = oculomotor_engine.evaluate_pupillometry(0.52)
    assert p_dilated.dilation_change_pct > 25.0
    assert p_dilated.cognitive_pupillary_response > 0.8


def test_evaluate_gaze_fixation_and_saccade(oculomotor_engine):
    t0 = 100.0
    # Stationary gaze (fixation)
    g1 = oculomotor_engine.evaluate_gaze(yaw_deg=2.0, pitch_deg=1.0, timestamp=t0)
    assert not g1.is_saccade
    g2 = oculomotor_engine.evaluate_gaze(yaw_deg=2.5, pitch_deg=1.2, timestamp=t0 + 0.2)
    assert not g2.is_saccade
    assert 0.0 <= g2.screen_x <= 1.0

    # Rapid gaze displacement (saccadic jump: 30 degrees in 50ms = 600 deg/s)
    g_saccade = oculomotor_engine.evaluate_gaze(yaw_deg=32.5, pitch_deg=5.0, timestamp=t0 + 0.25)
    assert g_saccade.is_saccade is True
    assert g_saccade.saccade_velocity_deg_s > oculomotor_engine.saccade_velocity_threshold


def test_compute_cognitive_workload_tiers():
    # Low Workload scenario
    p_low = PupillometryMetrics(cognitive_pupillary_response=0.1)
    b_low = BlinkDynamics(perclos=0.04, blink_suppressed=False)
    g_low = GazeTelemetry(fixation_duration_ms=250.0, dispersion_area=0.3)
    wl_low = OculomotorEngine.compute_cognitive_workload(p_low, b_low, g_low, autonomic_strain=0.15)

    assert wl_low.tier == WorkloadTier.LOW_LOAD.value
    assert wl_low.workload_index < 0.28
    assert wl_low.nasa_tlx.mental_demand < 50.0

    # Overload scenario (high CPR + blink suppression + autonomic strain + tunneling)
    p_high = PupillometryMetrics(cognitive_pupillary_response=0.92)
    b_high = BlinkDynamics(perclos=0.22, blink_suppressed=True)
    g_high = GazeTelemetry(fixation_duration_ms=1800.0, dispersion_area=0.05)
    wl_overload = OculomotorEngine.compute_cognitive_workload(p_high, b_high, g_high, autonomic_strain=0.85)

    assert wl_overload.tier in [WorkloadTier.HIGH_EFFORT.value, WorkloadTier.COGNITIVE_OVERLOAD.value]
    assert wl_overload.workload_index > 0.60
    assert wl_overload.nasa_tlx.mental_demand > 70.0
    assert wl_overload.nasa_tlx.effort > 70.0
    assert len(wl_overload.contributing_factors) > 0


def test_process_frame_end_to_end(oculomotor_engine, synthetic_open_eye):
    snap = oculomotor_engine.process_frame(
        left_eye_pts=synthetic_open_eye,
        right_eye_pts=synthetic_open_eye,
        pir_sample=0.46,
        yaw_deg=5.0,
        pitch_deg=-2.0,
        autonomic_strain=0.40,
    )

    assert isinstance(snap, OculomotorSnapshot)
    assert snap.eyes_detected is True
    assert snap.pupillometry.pupil_diameter_ratio == 0.46
    assert snap.blinks.ear > 0.20
    assert 0.0 <= snap.workload.workload_index <= 1.0

    # Serialization test
    d = snap.to_dict()
    assert "pupillometry" in d
    assert "blinks" in d
    assert "gaze" in d
    assert "workload" in d
    assert "nasa_tlx" in d["workload"]
