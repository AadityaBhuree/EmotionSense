# EmotionSense — Codebase Intelligence & Architecture Memory

> **Status:** Sprint 3 Completed — Enterprise SQLite Persistence, Clinical PDF Export & Hardened CI/CD  
> **Brand & Project:** EmotionSense  
> **Owner:** Aditya Bhure (AadityaBhuree)  
> **Date:** September 2026  

---

## 1. Project Purpose & Executive Summary

### 1.1 What Business Problem is Solved?
Human communication consists of verbal, vocal (pitch, tone, pauses), and non-verbal (facial expressions, micro-gestures) signals. Conventional sentiment analysis tools analyze only static text, missing over 80% of human emotional context. 

**EmotionSense** is an enterprise-grade multimodal emotion recognition and sentiment intelligence platform built in **Pure Python**. It captures, fuses, and analyzes:
1. **Visual / Facial Expressions**: Real-time webcam/video streams detecting micro-expressions (happy, sad, surprised, angry, fearful, disgusted, neutral, contempt) and 468-point facial mesh action units.
2. **Acoustic / Speech Signals**: Voice pitch, jitter, shimmer, tempo, energy, and vocal intonation metrics.
3. **Text / Semantic Content**: Natural language sentiment, affective context nuance, intent extraction, and conversational trajectory dynamics.
4. **Speech-to-Text & Phonetic Prosody**: Synchronized acoustic-lexical alignment linking live spoken audio buffers to transcribed tokens.
5. **Multimodal Fusion**: Real-time Valence-Arousal-Dominance (VAD) coordinate mapping, Engagement Index, Fatigue Level, Attention Scores, and Affective Anomaly Sentinel detection.
6. **Microservice API**: Production-ready asynchronous FastAPI REST microservice and bi-directional WebSocket streaming gateways.
7. **Enterprise Persistence & Clinical Reporting**: Embedded SQLite storage engine with cascading metadata management, automated legacy JSON sync, and publication-quality Clinical PDF diagnostics via ReportLab.

### 1.2 Target Users & Personas
- **Interview & Talent Assessment Teams**: Evaluating candidate engagement, confidence, stress resilience, and authenticity.
- **Mental Health & Wellness Providers**: Longitudinal tracking of affective response patterns, hyper-arousal spikes, and valence crashes with formal clinical PDF records.
- **Customer Experience & Sales Teams**: Real-time sentiment cues, conversational trajectory escalation alerts, and empathy guidance.
- **Researchers & EdTech**: Monitoring student focus, cognitive overload, fatigue, and engagement levels during live sessions.

---

## 2. Technology Stack

| Domain | Technology / Library | Role & Rationale |
| :--- | :--- | :--- |
| **Framework & UI** | **Streamlit + Precision Neuro-Instrument CSS** | Reactive workstation with calibrated technical grid, Russell circumplex vector trails, and tactile telemetry consoles |
| **Microservice Backend** | **FastAPI + Uvicorn + WebSockets + HTTPX** | Asynchronous high-throughput REST API and bi-directional streaming endpoints for telemetry, live speech, and session persistence |
| **Real-Time Video/Audio Stream** | **streamlit-webrtc + PyAV (`av`) + WebRTC** | Low-latency bi-directional video and audio frame transformation and container demuxing |
| **Computer Vision** | **MediaPipe + OpenCV + NumPy** | 468-point 3D Face Mesh, Facial Action Units (AU), and Micro-expression Classifier |
| **Audio & Acoustics** | **Librosa + SoundFile + SciPy** | Acoustic prosody, fundamental frequency (F0 pitch), RMS energy, jitter & shimmer |
| **NLP & Deep Transformers** | **RoBERTa / GoEmotions + PyTorch + Lexical VAD** | Hybrid Ensemble Emotion Classifier blending rule-based emoji/negation precision with neural contextual embeddings |
| **Speech Processing** | **SpeechRecognition + Acoustic Buffering** | Chunked PCM audio stream transcription with phonetic word-level prosodic alignment |
| **Data Visualization & Telemetry** | **Plotly (Graph Objects & Express)** | Dynamic Russell 2D Circumplex, Action Unit bar charts, radar dials, and real-time timelines |
| **Anomaly & Reporting** | **Statistical Z-Score Sentinel + Markdown/HTML** | Autonomous affective distress detection (valence crashes, hyper-arousal, attention collapse) and clinical reports |
| **Database & Persistence** | **SQLite3 (WAL Mode) + JSON/CSV** | ACID-compliant relational persistence for sessions, candidates, timeline samples, and affective anomaly history |
| **Clinical PDF Export** | **ReportLab** | Publication-grade printable PDF diagnostic evaluation dossiers with metadata blocks, VAD tables, and audit sign-off |
| **Code Quality & CI/CD** | **Ruff + Docker Buildx + GitHub Actions** | Lightning-fast static analysis, formatting, coverage reporting, and container build validation |

---

## 3. High-Level Architecture Diagram

```mermaid
flowchart TB
    subgraph Ingestion["Multimodal Ingestion Layer"]
        Cam["Webcam Video Stream (30 FPS)"]
        Mic["Microphone Audio Stream (16kHz PCM)"]
        Upload["File Container Upload (MP4 / WAV / MP3)"]
        TextIn["Direct Text / Transcript / Chat Stream"]
    end

    subgraph Streaming["Streaming & Communication Layer"]
        WebRTC["streamlit-webrtc Video/Audio Transformer"]
        FastAPI_WS["FastAPI WebSockets (/ws/stream-affect, /ws/stream-speech)"]
    end

    subgraph VisionEngine["Vision Intelligence Engine"]
        FaceMesh["MediaPipe Face Mesh (468 3D Landmarks)"]
        AUDetector["Facial Action Unit (AU) Extractor"]
        VisionEmotion["Micro-Expression Classifier\n(Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral)"]
        FaceMesh --> AUDetector --> VisionEmotion
    end

    subgraph AudioEngine["Acoustic & Speech Intelligence Engine"]
        Prosody["Pitch & Energy Estimator (F0, RMS)"]
        AcousticFeatures["Jitter, Shimmer, Harmonicity"]
        VocalTone["Vocal Emotion & Tone Classifier"]
        SpeechTranscriber["LiveSpeechTranscriber (Acoustic-Lexical Alignment)"]
        Prosody --> AcousticFeatures --> VocalTone
        Prosody --> SpeechTranscriber
    end

    subgraph TextEngine["Text Affect & NLP Engine"]
        Lexical["TextEmotionClassifier (VADER + Emoji + VAD Lexicon)"]
        Transformer["TransformerEmotionClassifier (RoBERTa-GoEmotions)"]
        Hybrid["HybridEmotionClassifier (Ensemble 55% Neural / 45% Lexical)"]
        Trajectory["ConversationAffectAnalyzer (Multi-Turn Dialogue Trajectory)"]
        Lexical --> Hybrid
        Transformer --> Hybrid
        Hybrid --> Trajectory
    end

    subgraph FusionEngine["Multimodal Fusion & Analytics"]
        TemporalAlign["Temporal Synchronization Window"]
        LateFusion["Weighted Multimodal Late Fusion"]
        AffectMetrics["Valence-Arousal-Dominance (VAD)\nEngagement, Fatigue, Attention Index"]
        Sentinel["AffectiveAnomalyDetector (Valence Crash, Hyper-Arousal, Distress)"]
        TemporalAlign --> LateFusion --> AffectMetrics --> Sentinel
    end

    subgraph StorageLayer["Persistence & Document Engine"]
        SQLiteDB["SQLite SessionDatabase (WAL Mode)\n(sessions, session_metadata, session_samples, session_anomalies)"]
        PDFGen["ClinicalPDFExporter (ReportLab Printable Reports)"]
        JSONSync["Legacy JSON Migration Engine"]
        JSONSync --> SQLiteDB
    end

    subgraph Interfaces["Presentation & Delivery Layer"]
        StreamlitUI["Streamlit Precision Neuro-Instrument UI"]
        FastAPI_REST["FastAPI Endpoints (/api/predict, /api/sessions, /api/stats)"]
        ReportExporter["DiagnosticReportGenerator (HTML / Markdown)"]
    end

    Cam --> WebRTC --> VisionEngine --> TemporalAlign
    Mic --> WebRTC --> AudioEngine --> TemporalAlign
    Mic --> FastAPI_WS --> SpeechTranscriber
    Upload --> VisionEngine & AudioEngine
    TextIn --> TextEngine --> TemporalAlign
    SpeechTranscriber --> TextEngine
    Sentinel --> StreamlitUI & FastAPI_REST & ReportExporter
    TemporalAlign --> StorageLayer
    StorageLayer --> StreamlitUI & FastAPI_REST & PDFGen
```

---

## 4. Repository Structure

```text
EmotionSense/
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI with Ruff, Pytest-cov & Docker build
├── .gitignore                         # Git exclusion rules (venv, *.db, data/, cache)
├── requirements.txt                   # Production dependencies (Streamlit, FastAPI, WebSockets, AV, ReportLab, etc.)
├── Dockerfile                         # Multi-stage production container definition
├── docker-compose.yml                 # Docker service orchestration
├── README.md                          # Repository overview and setup instructions
├── memory.md                          # System memory, architecture, and sprint logs
├── config.py                          # Global application tokens, paths (DB_PATH), and theme palette
├── app.py                             # Streamlit entry point, telemetry HUD, and navigation
├── server.py                          # FastAPI REST microservice and WebSocket streaming server
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                  # Thresholds, modality weights, and taxonomy definitions
│   │   └── types.py                   # Pydantic & dataclass definitions (AffectVector, SessionRecord, etc.)
│   ├── storage/
│   │   ├── __init__.py                # Exported storage engine and models
│   │   ├── models.py                  # AssessmentType, SessionMetadata, StoredSession dataclasses
│   │   └── db.py                      # SQLite SessionDatabase engine with WAL mode, CRUD & migrations
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── face_mesh.py               # 468-point MediaPipe face mesh processor
│   │   └── emotion_classifier.py      # Action Unit geometric heuristics & expression classifier
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── prosody.py                 # F0 pitch, RMS energy, jitter, and shimmer extraction
│   │   ├── voice_sentiment.py         # Acoustic sentiment and vocal affect classifier
│   │   └── speech_transcriber.py      # Live speech recognition & phonetic prosody alignment
│   ├── text/
│   │   ├── __init__.py                # Exported text classifiers and conversation analyzers
│   │   ├── nlp_emotion.py             # Lexical VAD, emoji dictionary, and negation parser
│   │   ├── transformer_emotion.py     # Deep learning RoBERTa-GoEmotions classifier with SEH protection
│   │   ├── hybrid_classifier.py       # Ensemble neural-lexical blender (55% neural / 45% lexical)
│   │   └── conversation_analyzer.py   # Multi-turn conversational trajectory & empathy engine
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── multimodal_fusion.py       # Tri-modal late fusion engine
│   │   ├── metrics.py                 # Engagement, Attention, and Fatigue indices
│   │   └── anomaly_detector.py        # Affective anomaly & emotional distress sentinel
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── styles.py                  # Neuro-instrument CSS, glassmorphic HUD styling
│   │   ├── charts.py                  # Plotly Russell Circumplex, AU spectrum, and radar charts
│   │   ├── components.py              # Telemetry badges, tactile status dials, and gauges
│   │   └── video_processor.py         # WebRTC VideoTransformer with landmark overlay
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                  # Structured application logging
│   │   ├── session_manager.py         # Session recording, persistence & JSON export
│   │   ├── report_generator.py        # Diagnostic HTML & Markdown clinical report generator
│   │   ├── pdf_exporter.py            # Publication-grade Clinical PDF report exporter via ReportLab
│   │   ├── demuxer.py                 # Audiovisual container demuxer (PyAV)
│   │   └── ws_client.py               # Asynchronous WebSocket client for affect and speech streams
│   └── pages/
│       ├── 1_💬_Text_Studio.py          # Single message, multi-turn chat & batch analysis
│       ├── 2_🎥_Live_Studio.py          # Real-time multimodal streaming studio
│       ├── 3_📁_File_Analysis.py        # Offline video & audio file container analysis
│       ├── 4_📑_Session_History.py      # Timeline scrubbing, SQLite persistence, metadata editor & PDF export
│       └── 5_📐_Architecture.py         # System architecture & multimodal documentation
├── scripts/
│   └── ws_stream_client.py            # CLI utility for real-time WebSocket telemetry testing
└── tests/
    ├── __init__.py
    ├── test_vision.py                 # MediaPipe landmarks, AU boundaries & null frame handling
    ├── test_audio.py                  # Pitch estimation, silence handling, voice sentiment
    ├── test_text_emotion.py           # Lexical VAD, emoji extraction, negation handling
    ├── test_transformer_hybrid.py     # Transformer availability, mode switching, ensemble fusion
    ├── test_speech_transcriber.py     # Audio buffer processing, speech transcription stream
    ├── test_demuxer.py                # Audiovisual demuxing, synchronized timeline & late fusion
    ├── test_webrtc_stream.py          # WebRTC audio resampling, prosody, transcript & HUD subtitles
    ├── test_ws_client.py              # WebSocket streaming client tests (URL construction, flow)
    ├── test_fusion.py                 # Multimodal late fusion, attention penalty, VAD quadrant
    ├── test_anomaly_detector.py       # Valence crash, hyper-arousal, sustained distress, fatigue
    ├── test_report_generator.py       # Markdown and HTML clinical diagnostic report formatting
    ├── test_storage.py                # SQLite schema, CRUD, cascade delete, JSON migration & stats
    ├── test_pdf_exporter.py           # ReportLab compilation, magic bytes, filesystem save & empty sessions
    └── test_api_server.py             # FastAPI REST endpoints, session CRUD & WebSocket streaming
```

---

## 5. API Endpoints Reference (`server.py`)

### 5.1 REST Endpoints

| Method | Path | Description | Request Body | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Microservice health check & engine readiness | None | `{"status": "healthy", "service": "EmotionSense API", "version": "2.0.0"}` |
| `POST` | `/api/predict/text` | Single text message affect & VAD prediction | `SingleTextRequest` (`text`, `mode`) | `TextEmotionResult` |
| `POST` | `/api/analyze/dialogue` | Multi-turn conversational trajectory & empathy advice | `DialogueTranscriptRequest` (`transcript`) | `DialogueEmotionSummary` |
| `POST` | `/api/batch/predict` | Batch text list affect processing | `BatchTextRequest` (`texts`, `mode`) | `List[TextEmotionResult]` |
| `POST` | `/api/anomalies/detect` | Affective anomaly detection across session frames | `AnomalyDetectionRequest` (`history`, `sample_rate`) | `List[Dict[str, Any]]` |
| `POST` | `/api/reports/generate` | Automated clinical diagnostic report export | `ReportGenerationRequest` (`session_data`, `format`) | `{"format": "html", "report": "..."}` |
| `GET` | `/api/sessions` | Query persisted sessions with filters & pagination | Query: `assessment_type`, `tag`, `search`, `limit`, `offset` | `{"count": int, "sessions": [...]}` |
| `GET` | `/api/sessions/{id}` | Get complete session details with timeline samples | Query: `include_samples` | `StoredSession` dictionary |
| `POST` | `/api/sessions` | Save or update session record in SQLite | `SaveSessionRequest` | `{"status": "success", "session_id": "..."}` |
| `PATCH` | `/api/sessions/{id}/metadata` | Update session clinical metadata, notes, and tags | `MetadataUpdateRequest` | `{"status": "success", "session_id": "..."}` |
| `DELETE` | `/api/sessions/{id}` | Delete session and cascading samples from database | None | `{"status": "success", "deleted_session_id": "..."}` |
| `POST` | `/api/sessions/migrate` | Trigger migration of legacy JSON sessions to SQLite | None | `{"status": "success", "migrated_sessions_count": int}` |
| `GET` | `/api/stats` | Retrieve platform-wide metrics and assessment type counts | None | `{"status": "success", "stats": {...}}` |

### 5.2 WebSocket Streaming Endpoints

| Protocol | Path | Description | Payload Format |
| :--- | :--- | :--- | :--- |
| `WS` | `/ws/stream-affect` | Real-time bi-directional affect streaming | JSON message: `{"text": "..."}` $\rightarrow$ JSON telemetry response |
| `WS` | `/ws/stream-speech` | Live audio streaming for transcription and phonetic affect | JSON chunk: `{"audio_chunk": "..."}` $\rightarrow$ JSON transcription + prosody response |

---

## 6. Verification & Quality Metrics

All **79** unit, integration, persistence, and streaming tests pass cleanly across Python 3.10+:

```bash
pytest -v
# ============================= 79 passed in 7.63s ==============================
```

- **Persistence & Database Suite (`test_storage.py`)**: Schema creation, table constraints, transaction rollbacks, index verification, session upsert, metadata updates, cascade deletion, legacy JSON file migration, and platform stats calculation.
- **Clinical PDF Exporter Suite (`test_pdf_exporter.py`)**: ReportLab document compilation, PDF magic bytes validation (`%PDF-`), multi-page layout generation, embedded VAD tables, anomaly blocks, clinical sign-off, filesystem saving, and empty session resilience.
- **Vision Suite (`test_vision.py`)**: MediaPipe landmark tolerances, AU bounds, null frame handling.
- **Audio Suite (`test_audio.py`)**: F0 pitch frequency estimation, silence thresholds, vocal sentiment.
- **Text & Transformer Suite (`test_text_emotion.py`, `test_transformer_hybrid.py`)**: Lexical VAD, emoji affect, negation inversions, protected PyTorch DLL SEH handling, hybrid ensemble blending.
- **Speech Transcriber Suite (`test_speech_transcriber.py`)**: Audio byte and numpy array transcription, mocked speech recognition, phonetic token alignment with acoustic pitch and energy, and offline fallback resilience.
- **Audiovisual Demuxer Suite (`test_demuxer.py`)**: Container demuxing, synthetic video/audio generation, corrupted input resilience, audio window extraction, synchronized timeline generation, and audiovisual late fusion.
- **WebRTC Stream Suite (`test_webrtc_stream.py`)**: Multimodal stream context thread safety, synthetic PyAV audio frame resampling, vocal prosody, autonomous speech recognition worker, live subtitle rendering, and tri-modal fusion.
- **WebSocket Streaming Suite (`test_ws_client.py`, `test_api_server.py`)**: URL construction, bidirectional `/ws/stream-affect` telemetry, and base64 audio chunk / JSON text `/ws/stream-speech` streaming.
- **Multimodal Fusion (`test_fusion.py`)**: Temporal alignment, attention penalty weighting, continuous 3D VAD mapping.
- **Sentinel Anomaly Suite (`test_anomaly_detector.py`)**: Valence crash, hyper-arousal spikes, sustained distress, cognitive fatigue overload.
- **Reporting Suite (`test_report_generator.py`)**: Clinical Markdown and responsive HTML diagnostic generation.
- **API Server Suite (`test_api_server.py`)**: FastAPI REST routes, full session persistence lifecycle (POST, GET, PATCH, DELETE, migrate, stats), and bidirectional WebSocket affect & speech streaming.
- **Static Code Analysis (`ruff check .`)**: Zero linting or formatting errors across entire repository.
