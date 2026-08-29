import numpy as np
from typing import Dict, List, Tuple, Optional
from src.core.types import FacialActionUnits, VisionEmotionResult
from src.core.config import LANDMARK_INDICES, EMOTION_LABELS


class FacialEmotionClassifier:
    """Classifies facial micro-expressions by extracting geometric FACS Action Units from landmarks."""

    def __init__(self):
        self.labels = EMOTION_LABELS

    def _euclidean_dist(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def extract_action_units(self, landmarks: np.ndarray) -> FacialActionUnits:
        """Computes geometric displacement metrics corresponding to FACS Action Units."""
        if landmarks is None or len(landmarks) < 468:
            return FacialActionUnits()

        # Face scale normalization factor (distance between outer eye corners)
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        face_scale = max(self._euclidean_dist(left_eye_outer, right_eye_outer), 1e-5)

        # 1. Inner Brow Raiser (AU1) - distance between inner brows and nose bridge
        inner_brow_left = landmarks[70]
        inner_brow_right = landmarks[336]
        nose_bridge = landmarks[168]
        au1_dist = (self._euclidean_dist(inner_brow_left, nose_bridge) + 
                    self._euclidean_dist(inner_brow_right, nose_bridge)) / (2.0 * face_scale)
        au1 = np.clip((au1_dist - 0.28) * 4.0, 0.0, 1.0)

        # 2. Brow Lowerer (AU4) - distance between left/right inner eyebrows
        brow_dist = self._euclidean_dist(inner_brow_left, inner_brow_right) / face_scale
        au4 = np.clip((0.35 - brow_dist) * 5.0, 0.0, 1.0)

        # 3. Cheek Raiser & Smile Lip Puller (AU6 / AU12)
        mouth_corner_left = landmarks[61]
        mouth_corner_right = landmarks[291]
        mouth_width = self._euclidean_dist(mouth_corner_left, mouth_corner_right) / face_scale
        upper_lip_center = landmarks[0]
        lower_lip_center = landmarks[17]
        
        # Smiling widens the mouth corners and pulls them upward
        au12 = np.clip((mouth_width - 0.50) * 4.5, 0.0, 1.0)
        au6 = np.clip(au12 * 0.85, 0.0, 1.0)

        # 4. Lip Corner Depressor (AU15 - Sadness)
        chin = landmarks[152]
        mouth_corner_y = (mouth_corner_left[1] + mouth_corner_right[1]) / 2.0
        lip_center_y = lower_lip_center[1]
        au15 = np.clip((mouth_corner_y - lip_center_y + 0.02) * 8.0, 0.0, 1.0)

        # 5. Upper Lid Raiser (AU5) & Eye Openness (EAR)
        ear_left = self._eye_aspect_ratio(landmarks, "left_eye", face_scale)
        ear_right = self._eye_aspect_ratio(landmarks, "right_eye", face_scale)
        avg_ear = (ear_left + ear_right) / 2.0
        au5 = np.clip((avg_ear - 0.22) * 5.0, 0.0, 1.0)

        # 6. Jaw Drop (AU26 - Surprise / Fear)
        mouth_openness = self._euclidean_dist(upper_lip_center, lower_lip_center) / face_scale
        au26 = np.clip((mouth_openness - 0.12) * 4.0, 0.0, 1.0)

        # Eye blinks
        blink_left = 1.0 if ear_left < 0.14 else 0.0
        blink_right = 1.0 if ear_right < 0.14 else 0.0

        return FacialActionUnits(
            inner_brow_raiser=float(au1),
            outer_brow_raiser=float(au1 * 0.9),
            brow_lowerer=float(au4),
            upper_lid_raiser=float(au5),
            cheek_raiser=float(au6),
            lid_tightener=float(au4 * 0.7),
            lip_corner_puller=float(au12),
            lip_corner_depressor=float(au15),
            lip_pressor=float(np.clip(au4 * 0.5 + (1.0 - mouth_openness * 3.0), 0.0, 1.0)),
            jaw_drop=float(au26),
            eye_blink_left=float(blink_left),
            eye_blink_right=float(blink_right),
        )

    def _eye_aspect_ratio(self, landmarks: np.ndarray, eye_key: str, face_scale: float) -> float:
        indices = LANDMARK_INDICES[eye_key]
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
        vertical_1 = self._euclidean_dist(p2, p6)
        vertical_2 = self._euclidean_dist(p3, p5)
        horizontal = max(self._euclidean_dist(p1, p4), 1e-5)
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def classify_emotion(
        self, landmarks: Optional[np.ndarray], head_pose: Dict[str, float]
    ) -> VisionEmotionResult:
        """Classifies micro-expressions into probability distribution over 8 canonical classes."""
        if landmarks is None or len(landmarks) < 468:
            return VisionEmotionResult(face_detected=False)

        aus = self.extract_action_units(landmarks)

        # Multi-factor FACS rule-based mapping with softmax normalization
        raw_scores = {
            "joy": aus.lip_corner_puller * 1.5 + aus.cheek_raiser * 1.0,
            "sadness": aus.lip_corner_depressor * 1.6 + aus.inner_brow_raiser * 0.8,
            "anger": aus.brow_lowerer * 1.7 + aus.lip_pressor * 0.8,
            "fear": aus.inner_brow_raiser * 1.2 + aus.upper_lid_raiser * 1.2 + aus.jaw_drop * 0.6,
            "surprise": aus.upper_lid_raiser * 1.4 + aus.jaw_drop * 1.5 + aus.outer_brow_raiser * 0.9,
            "disgust": aus.brow_lowerer * 0.9 + aus.lip_corner_depressor * 0.7,
            "neutral": 0.45,  # Baseline prior
            "contempt": aus.lip_corner_puller * 0.4 + aus.lip_corner_depressor * 0.4,
        }

        # Softmax normalization
        exp_scores = {k: np.exp(v * 2.5) for k, v in raw_scores.items()}
        total = sum(exp_scores.values())
        probabilities = {k: float(v / total) for k, v in exp_scores.items()}

        dominant_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[dominant_emotion]

        return VisionEmotionResult(
            face_detected=True,
            dominant_emotion=dominant_emotion,
            confidence=confidence,
            probabilities=probabilities,
            action_units=aus,
            head_pose=head_pose,
            landmarks_count=len(landmarks),
        )
