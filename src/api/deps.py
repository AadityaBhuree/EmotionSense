"""Shared service instances and dependencies for EmotionSense API Routers."""

from src.text import HybridEmotionClassifier, ConversationAffectAnalyzer
from src.audio.speech_transcriber import LiveSpeechTranscriber
from src.storage import SessionDatabase
from src.edge.runtime import ONNXEdgeInferenceEngine
from src.edge.quantizer import ModelQuantizationOptimizer
from src.edge.benchmark import EdgeBenchmarkSuite
from src.agent import ClinicalReasoningAgent, ClinicalChatCopilot
from src.analytics import (
    BiometricEngine,
    OculomotorEngine,
    SomatosensoryEngine,
    CredibilityEngine,
)
from src.analytics.forecasting import AffectiveHorizonForecaster

# Shared Singleton Engines
classifier = HybridEmotionClassifier(mode="hybrid")
conversation_analyzer = ConversationAffectAnalyzer(classifier)
speech_transcriber = LiveSpeechTranscriber(classifier.lexical_clf)
db = SessionDatabase()
edge_engine = ONNXEdgeInferenceEngine()
edge_quantizer = ModelQuantizationOptimizer()
edge_bench_suite = EdgeBenchmarkSuite(engine=edge_engine, quantizer=edge_quantizer)
clinical_agent = ClinicalReasoningAgent()
chat_copilot = ClinicalChatCopilot()
biometric_engine = BiometricEngine(fps=30.0)
oculomotor_engine = OculomotorEngine()
somatosensory_engine = SomatosensoryEngine()
credibility_engine = CredibilityEngine()
forecasting_engine = AffectiveHorizonForecaster()
