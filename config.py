"""Global configuration and design token constants for EmotionSense."""

import os
from pathlib import Path
from dataclasses import dataclass

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"

# Ensure data directories exist
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# App Metadata
APP_NAME = "EmotionSense"
APP_TAGLINE = "Enterprise Multimodal Affective Intelligence"
APP_ICON = "🎭"
VERSION = "1.0.0"

# Color Palette (Dark Cyber Affective Theme)
THEME_COLORS = {
    "background": "#080c14",
    "surface": "#0f172a",
    "surface_glass": "rgba(15, 23, 42, 0.75)",
    "border": "#1e293b",
    "border_glow": "rgba(99, 102, 241, 0.35)",
    "primary": "#6366f1",        # Indigo
    "primary_glow": "#818cf8",
    "secondary": "#06b6d4",      # Cyan
    "accent_joy": "#10b981",     # Emerald / Joy
    "accent_sad": "#3b82f6",     # Blue / Sadness
    "accent_anger": "#ef4444",   # Red / Anger
    "accent_fear": "#a855f7",    # Purple / Fear
    "accent_surprise": "#f59e0b",# Amber / Surprise
    "accent_disgust": "#14b8a6", # Teal / Disgust
    "accent_neutral": "#64748b", # Slate / Neutral
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
}

# Emotion Mappings & Hex Colors
EMOTION_COLORS = {
    "joy": "#10b981",
    "sadness": "#3b82f6",
    "anger": "#ef4444",
    "fear": "#a855f7",
    "surprise": "#f59e0b",
    "disgust": "#14b8a6",
    "neutral": "#64748b",
    "contempt": "#e11d48",
}

# Real-time Stream Parameters
STREAM_FPS = 30
FRAME_SAMPLE_RATE = 10  # sample 10 FPS for heavy inference if needed
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_DURATION_SEC = 0.5
TEMPORAL_WINDOW_SIZE = 15  # 15 frames rolling average
