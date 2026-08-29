"""Calculates higher-order behavioral metrics: Engagement Index, Attention Score, and Fatigue Level."""

import numpy as np
from typing import Dict, Any, Optional
from src.core.types import VisionEmotionResult, VoiceEmotionResult, FacialActionUnits, AcousticFeatures


class AffectMetricsCalculator:
    """Calculates real-time engagement, cognitive fatigue, and visual attention metrics."""

    @staticmethod
    def calculate_attention_score(vision: Optional[VisionEmotionResult]) -> float:
        """Estimates gaze / attention score based on head pose deviation from camera normal."""
        if not vision or not vision.face_detected:
            return 0.0

        yaw = abs(vision.head_pose.get("yaw", 0.0))
        pitch = abs(vision.head_pose.get("pitch", 0.0))
        roll = abs(vision.head_pose.get("roll", 0.0))

        # Deviation penalty: Looking directly at camera yields ~1.0
        yaw_penalty = min(yaw / 35.0, 1.0)
        pitch_penalty = min(pitch / 25.0, 1.0)
        roll_penalty = min(roll / 20.0, 1.0)

        composite_deviation = (yaw_penalty * 0.5 + pitch_penalty * 0.35 + roll_penalty * 0.15)
        attention = 1.0 - composite_deviation
        return float(np.clip(attention, 0.0, 1.0))

    @staticmethod
    def calculate_fatigue_level(
        vision: Optional[VisionEmotionResult],
        acoustics: Optional[AcousticFeatures] = None
    ) -> float:
        """Estimates cognitive/physical fatigue level based on eye closure, blink frequency & low vocal energy."""
        if not vision or not vision.face_detected:
            return 0.0

        aus = vision.action_units
        # Brow lowering / droop + eye blink duration
        blink_factor = (aus.eye_blink_left + aus.eye_blink_right) / 2.0
        droop_factor = aus.lip_corner_depressor * 0.4 + (1.0 - aus.upper_lid_raiser) * 0.3

        vocal_drag = 0.0
        if acoustics and acoustics.speech_active:
            if acoustics.tempo_bpm < 90 and acoustics.rms_energy < 0.02:
                vocal_drag = 0.3

        fatigue = blink_factor * 0.4 + droop_factor * 0.4 + vocal_drag * 0.2
        return float(np.clip(fatigue, 0.0, 1.0))

    @staticmethod
    def calculate_engagement_index(
        vision: Optional[VisionEmotionResult],
        voice: Optional[VoiceEmotionResult],
        attention_score: float
    ) -> float:
        """Calculates dynamic engagement index fusing eye contact, micro-expression intensity & vocal reactivity."""
        if not vision or not vision.face_detected:
            return 0.0

        # High arousal emotions (joy, surprise, focused concentration AU4) increase engagement
        probs = vision.probabilities
        expressive_arousal = (
            probs.get("joy", 0.0) * 1.2 +
            probs.get("surprise", 0.0) * 1.0 +
            probs.get("anger", 0.0) * 0.8 +
            vision.action_units.brow_lowerer * 0.6  # Deep concentration
        )

        vocal_engagement = 0.0
        if voice and voice.acoustics.speech_active:
            vocal_engagement = float(np.clip(voice.acoustics.rms_energy * 10.0 + (voice.acoustics.pitch_hz / 300.0) * 0.5, 0.0, 1.0))

        engagement = (
            attention_score * 0.45 +
            min(expressive_arousal, 1.0) * 0.35 +
            vocal_engagement * 0.20
        )
        return float(np.clip(engagement, 0.0, 1.0))
