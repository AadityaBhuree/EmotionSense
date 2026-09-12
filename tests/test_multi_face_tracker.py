"""Tests for MultiFaceTracker engine."""

import numpy as np
from src.vision.multi_face_tracker import MultiFaceTracker
from src.core.types import MultiFaceResult


def _create_synthetic_face(cx: float, cy: float, width: float = 0.2, height: float = 0.3) -> np.ndarray:
    """Generates a synthetic 468-point facial landmark array centered at (cx, cy)."""
    landmarks = np.zeros((468, 3), dtype=np.float32)
    # Scatter points within bounding box
    np.random.seed(42)
    landmarks[:, 0] = cx + (np.random.rand(468) - 0.5) * width
    landmarks[:, 1] = cy + (np.random.rand(468) - 0.5) * height
    landmarks[:, 2] = 0.0

    # Explicitly place critical indices so AU extractor doesn't divide by zero
    # Left/right eye outer corners (33, 263)
    landmarks[33] = [cx - width * 0.35, cy - height * 0.2, 0.0]
    landmarks[263] = [cx + width * 0.35, cy - height * 0.2, 0.0]
    # Eyebrows (70, 336, 168)
    landmarks[70] = [cx - width * 0.2, cy - height * 0.3, 0.0]
    landmarks[336] = [cx + width * 0.2, cy - height * 0.3, 0.0]
    landmarks[168] = [cx, cy - height * 0.15, 0.0]  # Nose bridge
    # Mouth (61, 291, 0, 17)
    landmarks[61] = [cx - width * 0.2, cy + height * 0.2, 0.0]
    landmarks[291] = [cx + width * 0.2, cy + height * 0.2, 0.0]
    landmarks[0] = [cx, cy + height * 0.15, 0.0]
    landmarks[17] = [cx, cy + height * 0.25, 0.0]

    return np.clip(landmarks, 0.0, 1.0)


def test_tracker_initialization():
    tracker = MultiFaceTracker(max_faces=4, max_disappeared=10)
    assert tracker.max_faces == 4
    assert tracker.max_disappeared == 10
    assert tracker.active_track_count == 0


def test_empty_frame_handling():
    tracker = MultiFaceTracker()
    res = tracker.track(None)
    assert isinstance(res, MultiFaceResult)
    assert res.face_count == 0
    assert len(res.faces) == 0

    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    res2 = tracker.track(empty_frame)
    assert res2.face_count == 0


def test_track_single_face_synthetic():
    tracker = MultiFaceTracker()
    face1 = _create_synthetic_face(0.3, 0.4)
    head_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    res = tracker.track_landmarks([(face1, head_pose)], (480, 640))
    assert res.face_count == 1
    assert len(res.faces) == 1
    assert res.faces[0].track_id == 0
    assert res.faces[0].label == "Participant A"
    assert res.faces[0].active_frames == 1
    assert res.faces[0].vision_result.face_detected is True


def test_track_two_faces_side_by_side():
    tracker = MultiFaceTracker(max_faces=4)
    face_left = _create_synthetic_face(0.25, 0.4)
    face_right = _create_synthetic_face(0.75, 0.4)
    head_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    res = tracker.track_landmarks([(face_left, head_pose), (face_right, head_pose)], (480, 640))
    assert res.face_count == 2
    assert tracker.active_track_count == 2
    track_ids = [f.track_id for f in res.faces]
    assert track_ids == [0, 1]
    labels = [f.label for f in res.faces]
    assert "Participant A" in labels
    assert "Participant B" in labels


def test_persistent_tracking_across_frames():
    tracker = MultiFaceTracker(max_faces=4)
    head_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    # Frame 1: Two faces at 0.25 and 0.75
    f1_left = _create_synthetic_face(0.25, 0.4)
    f1_right = _create_synthetic_face(0.75, 0.4)
    res1 = tracker.track_landmarks([(f1_left, head_pose), (f1_right, head_pose)], (480, 640))
    assert res1.face_count == 2

    # Frame 2: Faces shifted slightly to 0.27 and 0.73 (smooth motion)
    f2_left = _create_synthetic_face(0.27, 0.41)
    f2_right = _create_synthetic_face(0.73, 0.39)
    res2 = tracker.track_landmarks([(f2_left, head_pose), (f2_right, head_pose)], (480, 640))
    assert res2.face_count == 2

    # Track IDs should be preserved!
    assert res2.faces[0].track_id == 0
    assert res2.faces[1].track_id == 1
    assert res2.faces[0].active_frames == 2
    assert res2.faces[1].active_frames == 2


def test_disappearing_and_deregistration():
    tracker = MultiFaceTracker(max_faces=2, max_disappeared=3)
    head_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    face = _create_synthetic_face(0.5, 0.5)
    # Frame 1: face present
    tracker.track_landmarks([(face, head_pose)], (480, 640))
    assert tracker.active_track_count == 1

    # Frames 2, 3, 4: face disappeared
    tracker.track_landmarks([], (480, 640))
    tracker.track_landmarks([], (480, 640))
    tracker.track_landmarks([], (480, 640))
    # Frame 5: exceeds max_disappeared=3 -> purged
    res = tracker.track_landmarks([], (480, 640))
    assert res.face_count == 0
    assert tracker.active_track_count == 0


def test_overlay_rendering():
    tracker = MultiFaceTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    face = _create_synthetic_face(0.5, 0.5)
    res = tracker.track_landmarks([(face, {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})], (480, 640))

    annotated = tracker.draw_overlay(frame, res)
    assert annotated.shape == frame.shape
    # Frame should have had drawn lines, so sum of pixels > 0
    assert np.sum(annotated) > 0
