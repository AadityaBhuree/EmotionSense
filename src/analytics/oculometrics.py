"""Phase 11: Real-Time Oculomotor Telemetry & Cognitive Workload Engine.

Implements Eye Aspect Ratio (EAR), blink frequency & duration extraction,
PERCLOS drowsiness monitoring, Pupil-to-Iris Ratio (PIR), Cognitive Pupillary Response (CPR),
I-VT gaze fixation/saccade discrimination, and multi-sensor NASA-TLX Cognitive Workload fusion.
"""

import time
from typing import List, Tuple, Optional
import numpy as np

from src.core.cognitive_models import (
    WorkloadTier,
    PupillometryMetrics,
    BlinkDynamics,
    GazeTelemetry,
    NASATLXDimensions,
    CognitiveWorkloadRecord,
    OculomotorSnapshot,
)


class OculomotorEngine:
    """Enterprise computer vision and neurometric cognitive workload engine."""

    # Peri-ocular landmark indices for 468 MediaPipe Mesh
    LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    def __init__(
        self,
        ear_closed_threshold: float = 0.20,
        baseline_pir: float = 0.40,
        saccade_velocity_threshold: float = 120.0,
        buffer_window_sec: float = 60.0,
    ):
        self.ear_closed_threshold = ear_closed_threshold
        self.baseline_pir = baseline_pir
        self.saccade_velocity_threshold = saccade_velocity_threshold
        self.buffer_window_sec = buffer_window_sec

        # Temporal buffers for dynamics
        self.ear_buffer: List[Tuple[float, float]] = []          # (timestamp, ear)
        self.closure_history: List[Tuple[float, bool]] = []     # (timestamp, is_closed)
        self.gaze_history: List[Tuple[float, float, float]] = [] # (timestamp, yaw, pitch)
        self.pir_history: List[float] = []

        # Blink state machine
        self._is_currently_closed = False
        self._closure_start_time = 0.0
        self.logged_blinks: List[Tuple[float, float]] = []       # (timestamp, duration_ms)

    @staticmethod
    def calculate_ear(eye_points: np.ndarray) -> float:
        """Calculates Eye Aspect Ratio (EAR) from 6 2D/3D eye landmarks.
        
        EAR = (||p1 - p5|| + ||p2 - p4||) / (2 * ||p0 - p3||)
        """
        if len(eye_points) < 6:
            return 0.30

        pts = np.asarray(eye_points, dtype=np.float64)
        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        h = np.linalg.norm(pts[0] - pts[3]) + 1e-6

        ear = float((v1 + v2) / (2.0 * h))
        return float(np.clip(ear, 0.05, 0.60))

    def evaluate_blinks(
        self,
        left_ear: float,
        right_ear: float,
        timestamp: Optional[float] = None,
    ) -> BlinkDynamics:
        """Updates blink state machine, calculates PERCLOS and micro-sleep events."""
        if timestamp is None:
            timestamp = time.time()

        mean_ear = float((left_ear + right_ear) / 2.0)
        self.ear_buffer.append((timestamp, mean_ear))

        is_closed = mean_ear < self.ear_closed_threshold
        self.closure_history.append((timestamp, is_closed))

        # Evict old buffer points outside sliding window
        min_time = timestamp - self.buffer_window_sec
        self.ear_buffer = [(t, e) for t, e in self.ear_buffer if t >= min_time]
        self.closure_history = [(t, c) for t, c in self.closure_history if t >= min_time]
        self.logged_blinks = [(t, d) for t, d in self.logged_blinks if t >= min_time]

        # Blink state transitions
        micro_sleep = False
        if is_closed and not self._is_currently_closed:
            self._is_currently_closed = True
            self._closure_start_time = timestamp
        elif not is_closed and self._is_currently_closed:
            self._is_currently_closed = False
            duration_ms = (timestamp - self._closure_start_time) * 1000.0
            if 50.0 <= duration_ms <= 2500.0:
                self.logged_blinks.append((timestamp, duration_ms))
            if duration_ms > 500.0:
                micro_sleep = True
        elif is_closed and self._is_currently_closed:
            if (timestamp - self._closure_start_time) * 1000.0 > 500.0:
                micro_sleep = True

        # Compute PERCLOS (fraction of time closed)
        if len(self.closure_history) > 5:
            closed_count = sum(1 for _, c in self.closure_history if c)
            perclos = float(closed_count / len(self.closure_history))
        else:
            perclos = 0.05 if not is_closed else 0.50

        # Compute Blink Rate per minute
        if self.logged_blinks and len(self.ear_buffer) > 2:
            time_span_min = max(0.1, (timestamp - self.ear_buffer[0][0]) / 60.0)
            blink_rate = float(len(self.logged_blinks) / time_span_min)
            mean_dur = float(np.mean([d for _, d in self.logged_blinks]))
        else:
            blink_rate = 15.0
            mean_dur = 200.0

        blink_suppressed = blink_rate < 8.0 and not is_closed

        return BlinkDynamics(
            ear=round(mean_ear, 3),
            blink_rate_bpm=round(blink_rate, 1),
            mean_blink_duration_ms=round(mean_dur, 1),
            perclos=round(perclos, 3),
            micro_sleep_detected=micro_sleep,
            blink_suppressed=blink_suppressed,
        )

    def evaluate_pupillometry(
        self,
        pir_sample: float,
        timestamp: Optional[float] = None,
    ) -> PupillometryMetrics:
        """Evaluates Cognitive Pupillary Response (CPR) relative to calibrated baseline."""
        if timestamp is None:
            timestamp = time.time()

        clamped_pir = float(np.clip(pir_sample, 0.20, 0.75))
        self.pir_history.append(clamped_pir)
        if len(self.pir_history) > 300:
            self.pir_history.pop(0)

        # Baseline adjustment and relative change
        delta_pct = ((clamped_pir - self.baseline_pir) / self.baseline_pir) * 100.0
        # Positive dilation corresponds to cognitive effort (Locus Coeruleus activation)
        cpr = float(np.clip(delta_pct / 30.0, 0.0, 1.0))

        return PupillometryMetrics(
            pupil_diameter_ratio=round(clamped_pir, 3),
            baseline_ratio=round(self.baseline_pir, 3),
            dilation_change_pct=round(delta_pct, 1),
            cognitive_pupillary_response=round(cpr, 3),
            confidence=0.88,
            is_valid=True,
        )

    def evaluate_gaze(
        self,
        yaw_deg: float,
        pitch_deg: float,
        timestamp: Optional[float] = None,
    ) -> GazeTelemetry:
        """Estimates screen coordinates, fixation dwell duration, and saccadic velocity."""
        if timestamp is None:
            timestamp = time.time()

        yaw = float(np.clip(yaw_deg, -45.0, 45.0))
        pitch = float(np.clip(pitch_deg, -30.0, 30.0))

        screen_x = float(np.clip(0.5 + (yaw / 90.0), 0.02, 0.98))
        screen_y = float(np.clip(0.5 + (pitch / 60.0), 0.02, 0.98))

        saccade_vel = 0.0
        is_saccade = False
        fix_duration = 350.0

        if self.gaze_history:
            last_t, last_y, last_p = self.gaze_history[-1]
            dt = max(0.001, timestamp - last_t)
            angular_dist = np.sqrt((yaw - last_y) ** 2 + (pitch - last_p) ** 2)
            saccade_vel = float(angular_dist / dt)
            is_saccade = saccade_vel > self.saccade_velocity_threshold

            if not is_saccade:
                fix_duration = min(2000.0, (timestamp - self.gaze_history[0][0]) * 1000.0)
            else:
                fix_duration = 80.0

        self.gaze_history.append((timestamp, yaw, pitch))
        if len(self.gaze_history) > 60:
            self.gaze_history.pop(0)

        # Spatial dispersion
        if len(self.gaze_history) > 3:
            yaws = [y for _, y, _ in self.gaze_history]
            pitchs = [p for _, _, p in self.gaze_history]
            disp = float(np.sqrt(np.var(yaws) + np.var(pitchs)) / 25.0)
        else:
            disp = 0.10

        return GazeTelemetry(
            gaze_yaw_deg=round(yaw, 1),
            gaze_pitch_deg=round(pitch, 1),
            screen_x=round(screen_x, 3),
            screen_y=round(screen_y, 3),
            fixation_duration_ms=round(fix_duration, 1),
            saccade_velocity_deg_s=round(saccade_vel, 1),
            is_saccade=is_saccade,
            dispersion_area=round(float(np.clip(disp, 0.02, 1.0)), 3),
        )

    @staticmethod
    def compute_cognitive_workload(
        pupil: PupillometryMetrics,
        blinks: BlinkDynamics,
        gaze: GazeTelemetry,
        autonomic_strain: float = 0.30, # From Baevsky SI or HRV (0.0 to 1.0)
        speech_pause_ratio: float = 0.20, # From Audio Engine (0.0 to 1.0)
    ) -> CognitiveWorkloadRecord:
        """Multimodal late fusion of oculomotor biomarkers and autonomic strain into NASA-TLX workload."""
        cpr = pupil.cognitive_pupillary_response
        perclos = blinks.perclos
        suppress_weight = 0.15 if blinks.blink_suppressed else 0.0

        # Tunneling factor: high fixation duration and low dispersion
        tunneling = float(np.clip((gaze.fixation_duration_ms / 1000.0) * (1.0 - gaze.dispersion_area), 0.0, 1.0))

        # Composite Workload Index (0.0 to 1.0)
        workload_raw = (
            0.35 * cpr +
            0.20 * tunneling +
            0.15 * autonomic_strain +
            0.15 * suppress_weight +
            0.15 * speech_pause_ratio
        )
        workload_idx = float(np.clip(workload_raw, 0.05, 0.98))

        # NASA-TLX Dimensional Estimates
        mental_demand = float(np.clip(25.0 + cpr * 55.0 + tunneling * 20.0, 5.0, 99.0))
        temporal_demand = float(np.clip(20.0 + suppress_weight * 120.0 + autonomic_strain * 35.0, 5.0, 95.0))
        effort = float(np.clip(30.0 + cpr * 45.0 + autonomic_strain * 25.0, 10.0, 98.0))
        frustration = float(np.clip(15.0 + autonomic_strain * 50.0 + perclos * 35.0, 5.0, 95.0))

        # Mental Exhaustion / Fatigue Risk
        exhaustion_risk = float(np.clip(perclos * 1.8 + autonomic_strain * 0.4 + (workload_idx * 0.3), 0.05, 0.98))

        # Classification Tier
        if workload_idx < 0.25:
            tier = WorkloadTier.LOW_LOAD.value
        elif workload_idx < 0.60:
            tier = WorkloadTier.OPTIMAL_ENGAGEMENT.value
        elif workload_idx < 0.80:
            tier = WorkloadTier.HIGH_EFFORT.value
        else:
            tier = WorkloadTier.COGNITIVE_OVERLOAD.value

        factors = []
        if cpr > 0.45:
            factors.append("Task-Evoked Pupillary Dilation (CPR)")
        if blinks.blink_suppressed:
            factors.append("Intense Attentional Blink Suppression")
        if blinks.micro_sleep_detected or perclos > 0.15:
            factors.append("Elevated PERCLOS Drowsiness")
        if tunneling > 0.50:
            factors.append("Oculomotor Cognitive Tunneling")
        if autonomic_strain > 0.60:
            factors.append("Sympathetic Autonomic Nervous Strain")
        if not factors:
            factors = ["Balanced Cognitive Tone", "Normal Oculomotor Dispersion"]

        return CognitiveWorkloadRecord(
            workload_index=round(workload_idx, 3),
            tier=tier,
            nasa_tlx=NASATLXDimensions(
                mental_demand=round(mental_demand, 1),
                temporal_demand=round(temporal_demand, 1),
                effort=round(effort, 1),
                frustration=round(frustration, 1),
            ),
            mental_exhaustion_risk=round(exhaustion_risk, 3),
            contributing_factors=factors[:3],
        )

    def process_frame(
        self,
        left_eye_pts: np.ndarray,
        right_eye_pts: np.ndarray,
        pir_sample: float = 0.42,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        autonomic_strain: float = 0.30,
        timestamp: Optional[float] = None,
    ) -> OculomotorSnapshot:
        """Processes a single video frame's eye landmarks into a full OculomotorSnapshot."""
        if timestamp is None:
            timestamp = time.time()

        left_ear = self.calculate_ear(left_eye_pts)
        right_ear = self.calculate_ear(right_eye_pts)

        blinks = self.evaluate_blinks(left_ear, right_ear, timestamp=timestamp)
        pupil = self.evaluate_pupillometry(pir_sample, timestamp=timestamp)
        gaze = self.evaluate_gaze(yaw_deg, pitch_deg, timestamp=timestamp)
        workload = self.compute_cognitive_workload(pupil, blinks, gaze, autonomic_strain=autonomic_strain)

        return OculomotorSnapshot(
            timestamp=timestamp,
            pupillometry=pupil,
            blinks=blinks,
            gaze=gaze,
            workload=workload,
            eyes_detected=True,
        )
