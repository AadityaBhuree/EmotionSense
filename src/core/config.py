"""Internal model weights, emotion mapping matrices, and algorithm parameters."""

from typing import Dict, List
import numpy as np

# Emotion Classes
EMOTION_LABELS: List[str] = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "neutral",
    "contempt",
]

# Russell's Circumplex Affect Model Coordinates (Valence, Arousal, Dominance)
# [Valence (-1..1), Arousal (-1..1), Dominance (-1..1)]
EMOTION_VAD_COORDINATES: Dict[str, List[float]] = {
    "joy": [0.81, 0.51, 0.46],
    "sadness": [-0.63, -0.27, -0.33],
    "anger": [-0.51, 0.59, 0.25],
    "fear": [-0.64, 0.60, -0.43],
    "surprise": [0.40, 0.67, -0.13],
    "disgust": [-0.60, 0.35, 0.11],
    "neutral": [0.00, 0.00, 0.00],
    "contempt": [-0.55, 0.38, 0.30],
}

# Multimodal Fusion Modality Weights
DEFAULT_MODALITY_WEIGHTS = {
    "vision": 0.45,    # Visual facial expression weight
    "audio": 0.30,     # Acoustic prosody & tone weight
    "text": 0.35,      # NLP semantic affect weight
    "context": 0.10,   # Baseline / temporal prior weight
}

# Facial Action Unit (AU) Reference Landmark Indices in MediaPipe 468 Face Mesh
LANDMARK_INDICES = {
    "left_eye": [33, 160, 158, 133, 153, 144],
    "right_eye": [362, 385, 387, 263, 373, 380],
    "left_eyebrow": [70, 63, 105, 66, 107],
    "right_eyebrow": [336, 296, 334, 293, 300],
    "lips_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308],
    "lips_inner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415],
    "nose_tip": 1,
    "chin": 152,
    "forehead": 10,
    "left_cheek": 234,
    "right_cheek": 454,
}

# Normalization Thresholds for Acoustic Prosody
AUDIO_THRESHOLDS = {
    "pitch_min_hz": 65.0,     # Low male vocal range
    "pitch_max_hz": 400.0,    # High female vocal range
    "energy_silence_rms": 0.005,
    "jitter_stress_threshold": 0.025, # 2.5% jitter indicates vocal stress/tremor
    "shimmer_stress_threshold": 0.06,  # 6% shimmer indicates micro-tremors
}
