# EmotionSense 🎭🧠

> **Enterprise-Grade Real-Time Multimodal Emotion Recognition & Affective Intelligence Platform**

EmotionSense combines real-time computer vision, acoustic prosody analysis, and semantic sentiment intelligence to decode human emotion across visual, vocal, and textual modalities in a unified, high-performance **Pure Python Full-Stack** platform.

---

## 🚀 Key Features

- **👁️ 468-Point Vision Intelligence**: Real-time 3D facial landmark mesh, micro-expression classification, and Facial Action Units (AU) tracking.
- **🎙️ Acoustic Prosody & Tone Analysis**: Fundamental frequency (F0 pitch), RMS energy, jitter, shimmer, intonation, and speech sentiment decoding.
- **⚡ Multimodal Fusion Engine**: Temporal synchronization and weighted late fusion mapping to 2D/3D Valence-Arousal-Dominance (VAD) space.
- **📈 Real-Time Affect Telemetry**: Interactive Plotly radar charts, circumplex affect quadrant, live stream timelines, and attention/engagement/fatigue gauges.
- **📊 Session Intelligence & Export**: Session recording, timeline scrubbing, detailed affective reports, and CSV/JSON data export.

---

## 🛠️ Tech Stack

- **UI & Dashboard**: Streamlit, Custom Dark Glassmorphism CSS, Plotly
- **Real-Time Streaming**: `streamlit-webrtc`, PyAV
- **Computer Vision**: OpenCV, MediaPipe Face Mesh, NumPy
- **Audio & Acoustics**: Librosa, SoundFile, SciPy
- **Multimodal Intelligence**: Custom Late Temporal Fusion Engine

---

## 🏁 Quickstart

```bash
# 1. Clone repository
git clone https://github.com/AadityaBhuree/EmotionSense.git
cd EmotionSense

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Application
streamlit run app.py
```

---

## 📖 Architecture & Documentation

For in-depth architecture diagrams, data flow models, and roadmap details, see [memory.md](memory.md).
