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

# Color Palette (Precision Neuro-Affective Instrument Palette)
THEME_COLORS = {
    # Structural Canvas & Console Surfaces
    "background": "#0b0e14",         # Deep Obsidian Charcoal
    "surface": "#121622",            # Matte Instrument Panel
    "surface_raised": "#171d2b",     # Elevated Control Tile
    "surface_subtle": "#1c2334",     # Micro-interactive Hover
    "border": "#212a3d",             # Crisp 1px Precision Hairline
    "border_active": "#3b82f6",      # Active Focus Rail
    "border_subtle": "rgba(255, 255, 255, 0.07)",
    
    # Primary & Telemetry Signals
    "primary": "#3b82f6",            # Precision Cobalt Blue
    "primary_glow": "rgba(59, 130, 246, 0.25)",
    "secondary": "#0ea5e9",          # Cyan Telemetry Vector
    "accent_telemetry": "#10b981",   # Status Nominal Green
    "accent_warning": "#f59e0b",     # Caution Amber
    "accent_critical": "#ef4444",    # Anomaly / Escalation Vermilion
    
    # Typography & Hierarchy
    "text_primary": "#f1f5f9",       # High-Contrast Technical White
    "text_secondary": "#94a3b8",     # Slate Telemetry Secondary
    "text_muted": "#526079",         # Monospace Grid Inactive
    
    # Affect Spectrum Accents
    "accent_joy": "#10b981",
    "accent_sad": "#3b82f6",
    "accent_anger": "#ef4444",
    "accent_fear": "#a855f7",
    "accent_surprise": "#f59e0b",
    "accent_disgust": "#0d9488",
    "accent_neutral": "#64748b",
}

# Calibrated 8-Emotion Spectrum (Russell Circumplex & Plutchik Grounded)
EMOTION_COLORS = {
    "joy": "#10b981",       # High Valence, Mid-High Arousal (Vivid Emerald)
    "sadness": "#3b82f6",   # Low Valence, Low Arousal (Cobalt Slate)
    "anger": "#ef4444",     # Low Valence, High Arousal (Vermilion Red)
    "fear": "#a855f7",      # Low Valence, High Arousal (Amethyst Purple)
    "surprise": "#f59e0b",  # Mid-High Valence, High Arousal (Amber Gold)
    "disgust": "#0d9488",   # Low Valence, Low-Mid Arousal (Mineral Jade)
    "neutral": "#64748b",   # Baseline Equilibrium (Balanced Cool Slate)
    "contempt": "#e11d48",  # Asymmetric Negative (Crimson Rose)
}

# Real-time Stream Parameters
STREAM_FPS = 30
FRAME_SAMPLE_RATE = 10  # sample 10 FPS for heavy inference if needed
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_DURATION_SEC = 0.5
TEMPORAL_WINDOW_SIZE = 15  # 15 frames rolling average
