"""Page 5: Multimodal Architecture, Affect Theory & System Documentation."""

import streamlit as st
import pandas as pd

from src.ui.styles import inject_modern_styles
from src.ui.components import render_header
from src.core.config import EMOTION_VAD_COORDINATES

st.set_page_config(page_title="Architecture & Docs | EmotionSense", page_icon="📖", layout="wide")
inject_modern_styles()

render_header("System Architecture & Engineering Specs", "Mathematical Models, Temporal Late Fusion, WebRTC & Microservices")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏛️ Tri-Modal Fusion Pipeline",
    "💬 Conversational NLP & Escalation Math",
    "📐 Russell's Circumplex & 3D VAD",
    "🧬 FACS Action Units & Prosody",
    "🔌 REST & WebSocket API Specs",
    "🎞️ Audiovisual Demuxer & Sync",
    "🛡️ Anomaly Sentinel & Reports"
])

with tab1:
    st.markdown("""
    <div class="es-panel">
        <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.35rem;">Tri-Modal Temporal Late Fusion Framework</div>
        <p style="color: var(--text-sub); font-size: 0.85rem; line-height: 1.5; margin: 0;">
            EmotionSense dynamically synchronizes visual micro-expressions (MediaPipe 468), acoustic prosody (F0/Jitter/Shimmer), and lexical semantic NLP across a sliding temporal window (W = 15..30 frames).
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
        {"AU": "AU01", "Name": "Inner Brow Raiser", "Muscles": "Frontalis, pars medialis", "Key Indications": "Surprise, Fear, Sadness"},
        {"AU": "AU02", "Name": "Outer Brow Raiser", "Muscles": "Frontalis, pars lateralis", "Key Indications": "Surprise"},
        {"AU": "AU04", "Name": "Brow Lowerer", "Muscles": "Corrugator supercilii", "Key Indications": "Anger, Deep Concentration"},
        {"AU": "AU05", "Name": "Upper Lid Raiser", "Muscles": "Levator palpebrae superioris", "Key Indications": "Fear, Surprise, Excitement"},
        {"AU": "AU06", "Name": "Cheek Raiser", "Muscles": "Orbicularis oculi, pars orbitalis", "Key Indications": "Genuine Duchenne Smile / Joy"},
        {"AU": "AU12", "Name": "Lip Corner Puller", "Muscles": "Zygomaticus major", "Key Indications": "Joy, Happiness, Smiling"},
        {"AU": "AU15", "Name": "Lip Corner Depressor", "Muscles": "Depressor anguli oris", "Key Indications": "Sadness, Melancholy"},
        {"AU": "AU26", "Name": "Jaw Drop", "Muscles": "Masseter, Temporalis relaxed", "Key Indications": "Surprise, Shock"},
    ]
    st.dataframe(pd.DataFrame(facs_data), use_container_width=True)

with tab5:
    st.markdown("### 🔌 FastAPI Production Microservice Endpoints")
    st.markdown("EmotionSense provides high-throughput REST and real-time WebSocket endpoints configured in `src/api/server.py`:")

    st.code("""# Launch the FastAPI Microservice:
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# Endpoints:
GET  /health                                 -> Health and subsystem status
POST /api/v1/predict/text                    -> Single sentence emotion & 3D VAD
POST /api/v1/predict/batch                   -> High-throughput multi-text batch
POST /api/v1/analyze/dialogue                -> Multi-turn escalation & empathy analysis
POST /api/v1/anomalies/detect                -> Affective anomaly detection across timeline
POST /api/v1/reports/diagnostic              -> Automated HTML / Markdown clinical report
WS   /ws/affect-stream                       -> Real-time bidirectional streaming affect socket
WS   /ws/speech-stream                       -> Real-time audio PCM transcription & affect socket
""", language="bash")

    st.markdown("#### WebSocket Streaming Example")
    st.code("""import asyncio, websockets, json

async def stream():
    async with websockets.connect("ws://localhost:8000/ws/affect-stream") as ws:
        # Send raw frame or state
        await ws.send(json.dumps({"vision": {"dominant_emotion": "joy", "confidence": 0.95}}))
        response = await ws.recv()
        print("Fused State:", json.loads(response))

asyncio.run(stream())
""", language="python")

with tab6:
    st.markdown("### 🎞️ Audiovisual Container Demuxing & Dual-Track Sync")
    st.markdown("""
    When analyzing pre-recorded video media files (`.mp4`, `.avi`, `.mov`, `.mkv`), `AudiovisualDemuxer`
    (`src/media/demuxer.py`) uses PyAV container demuxing to decode video frames and resample audio into
    16kHz mono float32 arrays without external disk writes.
    """)
    st.markdown(r"""
    #### Temporal Window Matching
    For each video frame sampled at timestamp $t_k$:
    
    $$\text{Audio Window}(t_k) = \left[t_k - \frac{\Delta w}{2}, \, t_k + \frac{\Delta w}{2}\right]$$

    A rolling temporal window of $\Delta w = 1.0\text{s}$ centered at $t_k$ is extracted.
    Acoustic features (pitch $F_0$, RMS energy, jitter, vocal stress) and visual facial mesh Action Units are extracted in parallel and merged into a synchronized timeline point.
    """)

with tab7:
    st.markdown("### 🛡️ Affective Anomaly Sentinel & Clinical Report Generator")
    st.markdown("""
    The `AffectiveAnomalyDetector` continuously monitors affective trajectories to detect 5 critical behavioral anomaly patterns:
    """)
    st.markdown("""
    1. **Valence Crash**: Abrupt plunge in emotional valence ($\Delta \mathcal{V} \le -0.65$ within $\le 2.0\text{s}$). Indicates acute distress or panic trigger.
    2. **Hyper-Arousal Spike**: Extreme surge in emotional arousal ($\mathcal{A} \ge +0.75$ and $\Delta \mathcal{A} \ge +0.50$). Indicates aggression, fear shock, or rage.
    3. **Sustained Affective Distress**: Persistent negative valence ($\mathcal{V} < -0.35$ with distress emotions) for $> 5.0\text{s}$. Indicates clinical depressive episodes or chronic distress.
    4. **Cognitive Fatigue Overload**: Rising fatigue ($\ge 0.70$) accompanied by declining engagement ($\le 0.30$).
    5. **Attention Collapse**: Prolonged inattention (head pitch/yaw deviation with eye closure for $> 3.0\text{s}$).
    """)
    st.markdown("""
    Reports can be exported in both clinical HTML (with inline SVG charts, CSS styling, and alert tables) or GitHub Flavored Markdown formats via `DiagnosticReportGenerator`.
    """)
