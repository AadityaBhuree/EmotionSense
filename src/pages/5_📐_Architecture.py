"""Page 4: Multimodal Architecture, Affect Theory & FACS Specifications."""

import streamlit as st
import pandas as pd

from config import THEME_COLORS, EMOTION_COLORS
from src.ui.styles import inject_glassmorphic_styles
from src.ui.components import render_header, render_metric_card
from src.core.config import EMOTION_VAD_COORDINATES

st.set_page_config(page_title="Architecture & Docs | EmotionSense", page_icon="📖", layout="wide")
inject_glassmorphic_styles()

render_header("System Architecture & Theory", "Mathematical Models, Temporal Late Fusion & FACS Specifications")

tab1, tab2, tab3 = st.tabs(["🏛️ Multimodal Fusion Engine", "📐 Russell's Circumplex & VAD", "🧬 FACS Action Units"])

with tab1:
    st.markdown("""
    <div class="es-card">
        <h3 style="color: #f8fafc; margin-bottom: 0.5rem;">Weighted Temporal Late Fusion Pipeline</h3>
        <p style="color: #94a3b8;">
            EmotionSense synchronizes asynchronously sampled visual frames (30 FPS) and acoustic audio frames (16 kHz PCM) across a rolling sliding window (\(W=15..30\) frames).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(r"""
    #### Late Fusion Formulation

    For each discrete emotion category $e \in \mathcal{E} = \{\text{joy, sadness, anger, fear, surprise, disgust, neutral, contempt}\}$:

    $$P_{\text{fused}}(e, t) = w_v(t) \cdot P_{\text{vision}}(e, t) + w_a(t) \cdot P_{\text{audio}}(e, t) + w_c \cdot \bar{P}_{\text{history}}(e)$$

    Where:
    - $w_v(t), w_a(t)$ are dynamic confidence weights conditioned on face visibility and active speech detection ($VAD$).
    - $\bar{P}_{\text{history}}(e) = \frac{1}{K}\sum_{k=1}^K P_{\text{fused}}(e, t-k)$ provides temporal hysteresis and prevents abrupt flicker.
    - Continuous Affect projection: $\mathbf{A}(t) = \sum_{e \in \mathcal{E}} P_{\text{fused}}(e, t) \cdot \mathbf{v}_e^{\text{VAD}}$
    """)

with tab2:
    st.markdown("### 🌐 3D Valence-Arousal-Dominance Coordinate Space")
    st.markdown("Each emotion maps to a standardized coordinate in Russell's 3D continuous dimensional space:")

    vad_rows = []
    for emo, coords in EMOTION_VAD_COORDINATES.items():
        vad_rows.append({
            "Emotion": emo.capitalize(),
            "Valence (Unpleasant ◄► Pleasant)": f"{coords[0]:+.2f}",
            "Arousal (Calm ◄► Excited)": f"{coords[1]:+.2f}",
            "Dominance (Submissive ◄► Control)": f"{coords[2]:+.2f}",
        })
    st.dataframe(pd.DataFrame(vad_rows), use_container_width=True)

with tab3:
    st.markdown("### 🧬 Facial Action Coding System (FACS) Units")
    st.markdown("Extracted geometric displacement metrics via MediaPipe 468 3D landmark mesh:")

    facs_data = [
        {"AU": "AU1", "Name": "Inner Brow Raiser", "Muscles": "Frontalis, pars medialis", "Key Indications": "Surprise, Fear, Sadness"},
        {"AU": "AU2", "Name": "Outer Brow Raiser", "Muscles": "Frontalis, pars lateralis", "Key Indications": "Surprise"},
        {"AU": "AU4", "Name": "Brow Lowerer", "Muscles": "Corrugator supercilii", "Key Indications": "Anger, Deep Concentration"},
        {"AU": "AU5", "Name": "Upper Lid Raiser", "Muscles": "Levator palpebrae superioris", "Key Indications": "Fear, Surprise, Excitement"},
        {"AU": "AU6", "Name": "Cheek Raiser", "Muscles": "Orbicularis oculi, pars orbitalis", "Key Indications": "Genuine Duchenne Smile / Joy"},
        {"AU": "AU12", "Name": "Lip Corner Puller", "Muscles": "Zygomaticus major", "Key Indications": "Joy, Happiness, Smiling"},
        {"AU": "AU15", "Name": "Lip Corner Depressor", "Muscles": "Depressor anguli oris", "Key Indications": "Sadness, Melancholy"},
        {"AU": "AU26", "Name": "Jaw Drop", "Muscles": "Masseter, Temporalis relaxed", "Key Indications": "Surprise, Shock"},
    ]
    st.dataframe(pd.DataFrame(facs_data), use_container_width=True)
