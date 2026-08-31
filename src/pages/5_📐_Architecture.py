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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏛️ Tri-Modal Fusion Pipeline",
    "💬 Conversational NLP & Escalation Math",
    "📐 Russell's Circumplex & 3D VAD",
    "🧬 FACS Action Units & Prosody",
    "🔌 API & Integration Schemas"
])

with tab1:
    st.markdown("""
    <div class="es-card">
        <h3 style="color: #f8fafc; margin-bottom: 0.5rem;">Tri-Modal Temporal Late Fusion Framework</h3>
        <p style="color: #94a3b8;">
            EmotionSense dynamically synchronizes visual micro-expressions (MediaPipe 468), acoustic prosody (F0/Jitter/Shimmer), and lexical semantic NLP across a sliding temporal window (\(W = 15..30\) frames).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(r"""
    #### Mathematical Formulation

    For each discrete emotion category $e \in \mathcal{E} = \{\text{joy, sadness, anger, fear, surprise, disgust, neutral, contempt}\}$:

    $$P_{\text{fused}}(e, t) = w_v(t) \cdot P_{\text{vision}}(e, t) + w_a(t) \cdot P_{\text{audio}}(e, t) + w_t(t) \cdot P_{\text{text}}(e, t) + w_c \cdot \bar{P}_{\text{history}}(e)$$

    Where:
    - $w_v(t), w_a(t), w_t(t)$ are adaptive confidence weights conditioned on face landmark visibility, speech voice activity detection ($\text{VAD}_{\text{audio}}$), and text presence.
    - $\bar{P}_{\text{history}}(e) = \frac{1}{K}\sum_{k=1}^K P_{\text{fused}}(e, t-k)$ provides temporal hysteresis and prevents abrupt visual flicker.
    - Continuous Affect projection: $\mathbf{A}(t) = \sum_{e \in \mathcal{E}} P_{\text{fused}}(e, t) \cdot \mathbf{v}_e^{\text{VAD}}$ where $\mathbf{v}_e^{\text{VAD}} \in [-1, 1]^3$.
    """)

with tab2:
    st.markdown("### 💬 Conversational Dynamics & Affective Trajectory")
    st.markdown(r"""
    #### 1. Turn-by-Turn Escalation Risk Metric $E_k$
    The escalation risk at turn $k$ measures rising conflict or emotional distress:

    $$E_k = \min\left(1.0, \max\left(0.0, 1.3 P_k(\text{anger}) + 1.0 P_k(\text{fear}) + 0.9 P_k(\text{disgust}) + 1.1 P_k(\text{contempt}) + 0.4 \max(0, \mathcal{A}_k)\right)\right)$$

    #### 2. Empathy & Affective Synchrony Score $\rho_{\text{empathy}}$
    Measures emotional alignment between alternating speakers:

    $$\Delta \mathcal{V} = \frac{1}{N-1}\sum_{k=2}^N |\mathcal{V}_k - \mathcal{V}_{k-1}|, \quad \rho_{\text{empathy}} = \text{clip}\left(1.0 - 0.5 \cdot \Delta \mathcal{V}, 0.1, 1.0\right)$$
    """)

with tab3:
    st.markdown("### 🌐 3D Valence-Arousal-Dominance Coordinate Space")
    st.markdown("Standardized coordinates mapping discrete emotions into continuous 3D affect space:")

    vad_rows = []
    for emo, coords in EMOTION_VAD_COORDINATES.items():
        vad_rows.append({
            "Emotion": emo.capitalize(),
            "Valence (Unpleasant ◄► Pleasant)": f"{coords[0]:+.2f}",
            "Arousal (Calm ◄► Excited)": f"{coords[1]:+.2f}",
            "Dominance (Submissive ◄► Control)": f"{coords[2]:+.2f}",
        })
    st.dataframe(pd.DataFrame(vad_rows), use_container_width=True)

with tab4:
    st.markdown("### 🧬 Facial Action Coding System (FACS) Units")
    st.markdown("Geometric displacement vectors via MediaPipe 468 3D landmark mesh:")

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

with tab5:
    st.markdown("### 🔌 REST & Python Programmatic Integration")
    st.markdown("Quick integration snippet for integrating `EmotionSense` into backend pipelines:")
    st.code("""from src.text import TextEmotionClassifier, ConversationAffectAnalyzer

# 1. Single Text Message Affect Analysis
clf = TextEmotionClassifier()
res = clf.analyze_text("I am so thrilled and excited for our product release! 🎉")
print(res.dominant_emotion)  # 'joy'
print(res.affect.valence)    # +0.68
print(res.affect.arousal)    # +0.55

# 2. Conversational Transcript Trajectory
analyzer = ConversationAffectAnalyzer()
summary = analyzer.parse_and_analyze_transcript(\"\"\"
[10:00] User: I have an urgent issue!
[10:01] Agent: I am sorry to hear that, fixing it right now.
\"\"\")
print("Escalation Risk:", summary.escalation_risk)
print("Empathy Score:", summary.rapport_empathy_score)
""", language="python")


