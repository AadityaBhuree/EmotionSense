"""EmotionSense — Real-Time Multimodal Emotion Recognition & Affective Intelligence Platform."""

import streamlit as st
import time

# Page Configuration
st.set_page_config(
    page_title="EmotionSense | Multimodal Affective AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import APP_NAME, APP_TAGLINE, APP_ICON, THEME_COLORS
from src.ui.styles import inject_glassmorphic_styles
from src.ui.components import render_header, render_metric_card

# Apply custom dark glassmorphic styling
inject_glassmorphic_styles()

# Sidebar Navigation & Branding
with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem;">
        <span style="font-size: 2rem;">{APP_ICON}</span>
        <div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">{APP_NAME}</div>
            <div style="font-size: 0.75rem; color: #818cf8; font-weight: 600; text-transform: uppercase;">Multimodal Affect v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Platform Modules")
    st.info("👈 Use the navigation above or the quick launch cards below to enter each studio module.")

    st.markdown("---")
    st.markdown("### ⚡ Live Status")
    st.markdown("""
    - **Vision Engine**: `MediaPipe 468 FaceMesh` ✅
    - **Audio Engine**: `Acoustic Prosody (F0/Jitter)` ✅
    - **Fusion Engine**: `Temporal Late VAD Fusion` ✅
    """)

# Main Landing Page Content
render_header()

# Hero Banner
st.markdown("""
<div class="es-card" style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 27, 75, 0.4) 100%);">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
        <div style="max-width: 680px;">
            <h2 style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem;">
                Decode Human Affect Across Face, Voice & Emotion Dimensions
            </h2>
            <p style="color: #94a3b8; font-size: 1rem; line-height: 1.6;">
                EmotionSense bridges visual micro-expressions, vocal intonation, and continuous 3D Valence-Arousal-Dominance (VAD) spaces in low-latency real-time streams.
            </p>
        </div>
        <div style="display: flex; gap: 12px;">
            <a href="/Live_Studio" target="_self" style="text-decoration: none;">
                <div style="background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; padding: 12px 24px; border-radius: 12px; font-weight: 600; box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);">
                    🚀 Launch Live Studio
                </div>
            </a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Key Platform Metric Highlights
st.markdown("### 📊 Enterprise Telemetry Overview")
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Landmarks Tracked", "468 pts", delta="3D Face Mesh", color="#6366f1")
with c2:
    render_metric_card("Modalities Fused", "3 Inputs", delta="Vision + Audio + VAD", color="#06b6d4")
with c3:
    render_metric_card("Acoustic Features", "6 Prosodies", delta="F0, Jitter, Shimmer", color="#10b981")
with c4:
    render_metric_card("Inference Latency", "< 35 ms", delta="Real-time Stream", color="#f59e0b")

st.markdown("<br>", unsafe_allow_html=True)

# Feature Studio Grid
st.markdown("### 🛠️ Interactive Exploration Modules")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="es-card" style="height: 100%;">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🎥</div>
        <h3 style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">1. Live Multimodal Studio</h3>
        <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5;">
            Stream live webcam & microphone feeds with real-time 3D landmark wireframes, 8-emotion polar radar dials, and 2D circumplex quadrant maps.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="es-card" style="height: 100%;">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">📁</div>
        <h3 style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">2. Video / Audio File Analysis</h3>
        <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5;">
            Upload pre-recorded media files (MP4, WAV, MP3) for full timeline breakdown, frame-by-frame affect inspection, and acoustic stress scoring.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="es-card" style="height: 100%;">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">📑</div>
        <h3 style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">3. Session History & Reports</h3>
        <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5;">
            Scrub through historical sessions, inspect key emotional turning points, and export full telemetry datasets in CSV/JSON formats.
        </p>
    </div>
    """, unsafe_allow_html=True)
