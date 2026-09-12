"""Multi-Face Detection, Persistent Centroid Tracking, and Dyadic Affect Analysis Engine.

Tracks multiple faces across video frames, maintaining persistent spatial IDs (e.g. Participant A,
Participant B), calculating per-face FACS Action Units and continuous affect (Valence-Arousal).
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.core.types import MultiFaceResult, TrackedFace
from src.vision.emotion_classifier import FacialEmotionClassifier
from src.vision.face_mesh import FaceMeshDetector


class MultiFaceTracker:
    """Tracks multiple faces across frames with persistent IDs and per-face affect."""

    DEFAULT_LABELS = ["Participant A", "Participant B", "Participant C", "Participant D"]

    def __init__(
        self,
        max_faces: int = 4,
        max_disappeared: int = 15,
        max_distance: float = 0.35,
        detector: Optional[FaceMeshDetector] = None,
        classifier: Optional[FacialEmotionClassifier] = None,
    ):
        """Initializes the multi-face tracker.

        Args:
            max_faces: Maximum number of faces to simultaneously track.
            max_disappeared: Number of consecutive frames a face can disappear before deregistration.
            max_distance: Maximum normalized distance (0.0 - 1.0) allowed for centroid re-identification.
            detector: Optional FaceMeshDetector instance.
            classifier: Optional FacialEmotionClassifier instance.
        """
        self.max_faces = max_faces
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.detector = detector or FaceMeshDetector(max_num_faces=max_faces)
        self.classifier = classifier or FacialEmotionClassifier()

        self._next_track_id: int = 0
        self._tracks: Dict[int, TrackedFace] = {}

    def reset(self):
        """Resets all tracking states and resets track ID sequence."""
        self._next_track_id = 0
        self._tracks.clear()

    @property
    def active_track_count(self) -> int:
        """Returns the number of currently active face tracks."""
        return len(self._tracks)

    def _compute_bbox_and_centroid(
        self, landmarks: np.ndarray, frame_w: int, frame_h: int
    ) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
        """Computes pixel bounding box (x, y, w, h) and normalized centroid (cx, cy)."""
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]

        min_x = float(np.clip(np.min(xs), 0.0, 1.0))
        max_x = float(np.clip(np.max(xs), 0.0, 1.0))
        min_y = float(np.clip(np.min(ys), 0.0, 1.0))
        max_y = float(np.clip(np.max(ys), 0.0, 1.0))

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        px_x = int(min_x * frame_w)
        px_y = int(min_y * frame_h)
        px_w = max(1, int((max_x - min_x) * frame_w))
        px_h = max(1, int((max_y - min_y) * frame_h))

        return (px_x, px_y, px_w, px_h), (cx, cy)

    def track(self, frame: np.ndarray) -> MultiFaceResult:
        """Processes an image frame, detects faces, updates tracking, and returns affect."""
        if frame is None or frame.size == 0:
            return self._handle_empty_detections()

        h, w = frame.shape[:2]
        detected_faces = self.detector.process_all_faces(frame)
        return self.track_landmarks(detected_faces, (h, w))

    def track_landmarks(
        self,
        detected_faces: List[Tuple[np.ndarray, Dict[str, float]]],
        frame_shape: Tuple[int, int],
    ) -> MultiFaceResult:
        """Tracks faces given pre-extracted landmarks and head poses."""
        h, w = frame_shape
        if not detected_faces:
            return self._handle_empty_detections()

        # 1. Parse current frame detections
        detections: List[Dict] = []
        for landmarks, head_pose in detected_faces:
            bbox, centroid = self._compute_bbox_and_centroid(landmarks, w, h)
            emotion_res = self.classifier.classify_emotion(landmarks, head_pose)
            detections.append({
                "bbox": bbox,
                "centroid": centroid,
                "landmarks": landmarks,
                "head_pose": head_pose,
                "vision_result": emotion_res,
            })

        # 2. Match detections to existing tracks
        if not self._tracks:
            for det in detections:
                self._register_track(det)
        else:
            self._update_tracks(detections)

        # 3. Compile active tracked faces sorted by track_id
        active_faces = [
            t for t in self._tracks.values() if t.lost_frames == 0
        ]
        active_faces.sort(key=lambda t: t.track_id)

        return MultiFaceResult(
            faces=active_faces,
            face_count=len(active_faces),
            timestamp=time.time(),
        )

    def _register_track(self, detection: Dict):
        """Registers a new face track."""
        track_id = self._next_track_id
        self._next_track_id += 1
        label = (
            self.DEFAULT_LABELS[track_id % len(self.DEFAULT_LABELS)]
            if track_id < len(self.DEFAULT_LABELS)
            else f"Subject {track_id + 1}"
        )
        self._tracks[track_id] = TrackedFace(
            track_id=track_id,
            bbox=detection["bbox"],
            centroid=detection["centroid"],
            vision_result=detection["vision_result"],
            active_frames=1,
            lost_frames=0,
            label=label,
        )

    def _update_tracks(self, detections: List[Dict]):
        """Associates new detections with existing tracks using centroid distance."""
        track_ids = list(self._tracks.keys())
        track_centroids = np.array([self._tracks[tid].centroid for tid in track_ids])
        det_centroids = np.array([d["centroid"] for d in detections])

        # Distance matrix: (N_tracks, M_detections)
        diff = track_centroids[:, np.newaxis, :] - det_centroids[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        matched_tracks = set()
        matched_detections = set()

        # Greedy match by lowest Euclidean distance
        flat_indices = np.argsort(distances, axis=None)
        for idx in flat_indices:
            row = int(idx // distances.shape[1])
            col = int(idx % distances.shape[1])

            if row in matched_tracks or col in matched_detections:
                continue

            if distances[row, col] <= self.max_distance:
                matched_tracks.add(row)
                matched_detections.add(col)
                track_id = track_ids[row]
                det = detections[col]

                # Update existing track
                track = self._tracks[track_id]
                track.bbox = det["bbox"]
                track.centroid = det["centroid"]
                track.vision_result = det["vision_result"]
                track.active_frames += 1
                track.lost_frames = 0

        # Unmatched tracks (increment lost_frames)
        for row, tid in enumerate(track_ids):
            if row not in matched_tracks:
                self._tracks[tid].lost_frames += 1

        # Unmatched detections (new tracks)
        for col, det in enumerate(detections):
            if col not in matched_detections and len(self._tracks) < self.max_faces:
                self._register_track(det)

        # Purge dead tracks exceeding max_disappeared
        dead_tracks = [
            tid for tid, t in self._tracks.items() if t.lost_frames > self.max_disappeared
        ]
        for tid in dead_tracks:
            del self._tracks[tid]

    def _handle_empty_detections(self) -> MultiFaceResult:
        """Handles a frame where no faces were detected."""
        dead_tracks = []
        for tid, t in self._tracks.items():
            t.lost_frames += 1
            if t.lost_frames > self.max_disappeared:
                dead_tracks.append(tid)

        for tid in dead_tracks:
            del self._tracks[tid]

        return MultiFaceResult(faces=[], face_count=0, timestamp=time.time())

    def draw_overlay(
        self,
        frame: np.ndarray,
        result: MultiFaceResult,
        show_affect: bool = True,
    ) -> np.ndarray:
        """Renders precision HUD bounding boxes and badges on video frame."""
        if frame is None or cv2 is None or not result.faces:
            return frame

        annotated = frame.copy()
        palette = [
            (59, 130, 246),   # Participant A - Blue
            (16, 185, 129),   # Participant B - Emerald
            (168, 85, 247),   # Participant C - Purple
            (245, 158, 11),   # Participant D - Amber
        ]

        for face in result.faces:
            color = palette[face.track_id % len(palette)]
            x, y, w, h = face.bbox

            # Draw sleek corner brackets
            corner_len = min(20, w // 4, h // 4)
            t = 2
            # Top-left
            cv2.line(annotated, (x, y), (x + corner_len, y), color, t)
            cv2.line(annotated, (x, y), (x, y + corner_len), color, t)
            # Top-right
            cv2.line(annotated, (x + w, y), (x + w - corner_len, y), color, t)
            cv2.line(annotated, (x + w, y), (x + w, y + corner_len), color, t)
            # Bottom-left
            cv2.line(annotated, (x, y + h), (x + corner_len, y + h), color, t)
            cv2.line(annotated, (x, y + h), (x, y + h - corner_len), color, t)
            # Bottom-right
            cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), color, t)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), color, t)

            # Badge banner
            label_text = f"{face.label} (ID {face.track_id})"
            emotion_text = f"{face.vision_result.dominant_emotion.upper()} {face.vision_result.confidence * 100:.0f}%"
            val_text = f"V: {face.vision_result.affect.valence:+.2f} | A: {face.vision_result.affect.arousal:+.2f}"

            badge_y = max(24, y - 8)
            cv2.putText(
                annotated,
                label_text,
                (x, badge_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            if show_affect:
                cv2.putText(
                    annotated,
                    f"{emotion_text} | {val_text}",
                    (x, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )

        return annotated
