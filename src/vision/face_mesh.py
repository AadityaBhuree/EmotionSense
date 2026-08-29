try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from src.core.config import LANDMARK_INDICES


class FaceMeshDetector:
    """Extracts 468 3D facial landmarks and calculates head pose from video frames."""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._mp_face_mesh = None
        self._face_mesh = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_drawing_styles = mp.solutions.drawing_styles
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                max_num_faces=self.max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except Exception as e:
            self._face_mesh = None

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """Processes an RGB frame and returns normalized landmarks (468, 3) and head pose."""
        if frame is None or self._face_mesh is None or cv2 is None:
            return None, {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.zeros((len(face_landmarks.landmark), 3), dtype=np.float32)

        for i, lm in enumerate(face_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]

        head_pose = self._estimate_head_pose(landmarks, w, h)
        return landmarks, head_pose

    def _estimate_head_pose(self, landmarks: np.ndarray, w: int, h: int) -> Dict[str, float]:
        """Estimates 3D Head orientation (Yaw, Pitch, Roll) using 2D-3D correspondence."""
        try:
            # 2D image points of key landmarks
            nose_tip = landmarks[LANDMARK_INDICES["nose_tip"]]
            chin = landmarks[LANDMARK_INDICES["chin"]]
            left_eye_outer = landmarks[33]
            right_eye_outer = landmarks[263]
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]

            image_points = np.array([
                [nose_tip[0] * w, nose_tip[1] * h],
                [chin[0] * w, chin[1] * h],
                [left_eye_outer[0] * w, left_eye_outer[1] * h],
                [right_eye_outer[0] * w, right_eye_outer[1] * h],
                [left_mouth[0] * w, left_mouth[1] * h],
                [right_mouth[0] * w, right_mouth[1] * h],
            ], dtype=np.float64)

            # 3D generic model points
            model_points = np.array([
                (0.0, 0.0, 0.0),            # Nose tip
                (0.0, -330.0, -65.0),       # Chin
                (-225.0, 170.0, -135.0),    # Left eye outer corner
                (225.0, 170.0, -135.0),     # Right eye outer corner
                (-150.0, -150.0, -125.0),   # Left Mouth corner
                (150.0, -150.0, -125.0)     # Right mouth corner
            ], dtype=np.float64)

            # Camera internals
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rot_vec, trans_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

            rot_mat, _ = cv2.Rodrigues(rot_vec)
            # Deconstruct angles
            sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
                yaw = np.arctan2(-rot_mat[2, 0], sy)
                roll = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
            else:
                pitch = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
                yaw = np.arctan2(-rot_mat[2, 0], sy)
                roll = 0.0

            return {
                "yaw": float(np.degrees(yaw)),
                "pitch": float(np.degrees(pitch)),
                "roll": float(np.degrees(roll)),
            }
        except Exception:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    def draw_mesh_overlay(self, frame: np.ndarray, landmarks: np.ndarray, color=(99, 238, 99)) -> np.ndarray:
        """Draws aesthetic facial wireframe points & contour connections."""
        if frame is None or landmarks is None:
            return frame

        h, w, _ = frame.shape
        annotated = frame.copy()

        # Draw key facial contours (lips, eyebrows, eyes)
        for key in ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "lips_outer"]:
            indices = LANDMARK_INDICES[key]
            pts = []
            for idx in indices:
                pt = (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h))
                pts.append(pt)
                cv2.circle(annotated, pt, 1, (99, 102, 241), -1)

            if len(pts) > 2:
                cv2.polylines(annotated, [np.array(pts, dtype=np.int32)], isClosed=True, color=color, thickness=1)

        # Highlight nose & chin anchors
        nose_pt = (int(landmarks[1][0] * w), int(landmarks[1][1] * h))
        chin_pt = (int(landmarks[152][0] * w), int(landmarks[152][1] * h))
        cv2.circle(annotated, nose_pt, 3, (6, 182, 212), -1)
        cv2.circle(annotated, chin_pt, 3, (244, 63, 94), -1)

        return annotated
