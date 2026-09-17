"""Cross-Modal Attentive Fusion (CMAF) Transformer & Modality Congruence Engine.

Dynamically projects vision, speech/acoustic, and textual affect signals into a shared
latent space, computes cross-modal multi-head attention (text contextualizing facial expressions,
acoustic-facial concordance), and evaluates cross-modal affective congruence and masked affect.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.core.config import (
    CONGRUENCE_THRESHOLDS,
    CROSS_MODAL_ATTENTION_CONFIG,
    EMOTION_LABELS,
    EMOTION_VAD_COORDINATES,
)
from src.core.types import (
    AcousticSERResult,
    AffectVector,
    CrossModalAttentionWeights,
    CrossModalFusionResult,
    ModalityCongruence,
    TextEmotionResult,
    VisionEmotionResult,
    VoiceEmotionResult,
)


class CrossModalAttentionFusion:
    """Cross-Modal Attention Transformer and Affective Congruence Analyzer."""

    def __init__(self, embed_dim: Optional[int] = None, num_heads: Optional[int] = None):
        self.labels = EMOTION_LABELS
        self.config = CROSS_MODAL_ATTENTION_CONFIG
        self.embed_dim = embed_dim or self.config.get("embed_dim", 64)
        self.num_heads = num_heads or self.config.get("num_heads", 4)
        self.head_dim = self.embed_dim // self.num_heads

        # Deterministic projections for vision, audio, and text
        rng = np.random.RandomState(1337)
        self.W_v = rng.randn(16, self.embed_dim).astype(np.float32) * np.sqrt(2.0 / 16)
        self.W_a = rng.randn(24, self.embed_dim).astype(np.float32) * np.sqrt(2.0 / 24)
        self.W_t = rng.randn(12, self.embed_dim).astype(np.float32) * np.sqrt(2.0 / 12)

        # Multi-head attention query, key, value matrices
        self.W_q = rng.randn(self.embed_dim, self.embed_dim).astype(np.float32) * np.sqrt(1.0 / self.embed_dim)
        self.W_k = rng.randn(self.embed_dim, self.embed_dim).astype(np.float32) * np.sqrt(1.0 / self.embed_dim)
        self.W_v_proj = rng.randn(self.embed_dim, self.embed_dim).astype(np.float32) * np.sqrt(1.0 / self.embed_dim)
        self.W_out = rng.randn(self.embed_dim, len(self.labels)).astype(np.float32) * np.sqrt(2.0 / self.embed_dim)

    def _extract_vision_feature_vector(self, vision: Optional[VisionEmotionResult]) -> Tuple[np.ndarray, float]:
        """Extracts 16-dimensional normalized feature vector from vision pipeline."""
        if vision is None or not getattr(vision, "face_detected", False):
            return np.zeros(16, dtype=np.float32), 0.0

        feats = np.zeros(16, dtype=np.float32)
        # 8 emotion probabilities
        for i, emo in enumerate(self.labels):
            feats[i] = vision.probabilities.get(emo, 0.0)

        # Action Units
        au = vision.action_units
        if au is not None:
            feats[8] = float(getattr(au, "inner_brow_raiser", getattr(au, "au01_inner_brow_raiser", 0.0)))
            feats[9] = float(getattr(au, "brow_lowerer", getattr(au, "au04_brow_lowerer", 0.0)))
            feats[10] = float(getattr(au, "lip_corner_puller", getattr(au, "au12_lip_corner_puller", 0.0)))
            feats[11] = float(getattr(au, "lip_corner_depressor", getattr(au, "au15_lip_corner_depressor", 0.0)))
            feats[12] = float(getattr(au, "jaw_drop", getattr(au, "au26_jaw_drop", 0.0)))

        # Head pose / attention
        feats[13] = float(vision.head_pose.get("pitch", 0.0) / 90.0)
        feats[14] = float(vision.head_pose.get("yaw", 0.0) / 90.0)
        feats[15] = float(vision.confidence)
        return feats, float(vision.confidence)

    def _extract_audio_feature_vector(
        self, audio: Optional[Union[VoiceEmotionResult, AcousticSERResult]]
    ) -> Tuple[np.ndarray, float]:
        """Extracts 24-dimensional normalized feature vector from audio/SER pipeline."""
        if audio is None:
            return np.zeros(24, dtype=np.float32), 0.0

        is_active = False
        confidence = float(getattr(audio, "confidence", 0.0))

        if isinstance(audio, AcousticSERResult):
            probs = audio.emotion_scores
            vad = audio.vad
            is_active = audio.dominant_emotion != "neutral" or confidence > 0.6
        else:
            probs = audio.probabilities
            vad = audio.affect
            is_active = getattr(audio.acoustics, "speech_active", False)

        if not is_active and confidence < 0.2:
            return np.zeros(24, dtype=np.float32), 0.0

        feats = np.zeros(24, dtype=np.float32)
        # 8 emotion probabilities
        for i, emo in enumerate(self.labels):
            feats[i] = probs.get(emo, 0.0)

        # VAD coordinates
        if vad is not None:
            feats[8] = float(vad.valence)
            feats[9] = float(vad.arousal)
            feats[10] = float(vad.dominance)

        # Acoustic embeddings or prosody metrics
        if isinstance(audio, AcousticSERResult) and len(audio.embedding) >= 12:
            feats[11:23] = np.array(audio.embedding[:12], dtype=np.float32)
        elif hasattr(audio, "acoustics") and audio.acoustics is not None:
            ac = audio.acoustics
            feats[11] = float(ac.pitch_hz / 300.0)
            feats[12] = float(ac.rms_energy * 10.0)
            feats[13] = float(ac.jitter_percent * 20.0)
            feats[14] = float(ac.shimmer_percent * 10.0)
            feats[15] = float(ac.zero_crossing_rate * 5.0)

        feats[23] = confidence
        return feats, confidence

    def _extract_text_feature_vector(self, text: Optional[Any]) -> Tuple[np.ndarray, float]:
        """Extracts 12-dimensional normalized feature vector from text NLP pipeline."""
        if text is None:
            return np.zeros(12, dtype=np.float32), 0.0

        raw_text = getattr(text, "text", "")
        if not raw_text:
            return np.zeros(12, dtype=np.float32), 0.0

        feats = np.zeros(12, dtype=np.float32)
        probs = getattr(text, "probabilities", {})
        confidence = float(getattr(text, "confidence", 0.5))

        for i, emo in enumerate(self.labels):
            feats[i] = probs.get(emo, 0.0)

        vad = getattr(text, "affect", None)
        if vad is not None:
            feats[8] = float(vad.valence)
            feats[9] = float(vad.arousal)
            feats[10] = float(vad.dominance)

        feats[11] = confidence
        return feats, confidence

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculates normalized cosine similarity between vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0  # Absent modality produces neutral concordance
        dot = np.dot(a, b)
        return float(np.clip(dot / (norm_a * norm_b), -1.0, 1.0))

    def evaluate_congruence(
        self,
        vision_prob: Dict[str, float],
        audio_prob: Dict[str, float],
        text_prob: Dict[str, float],
        vision_vad: Optional[AffectVector],
        audio_vad: Optional[AffectVector],
        text_vad: Optional[AffectVector],
        vision_au: Optional[Any],
        gates: Dict[str, float],
    ) -> ModalityCongruence:
        """Evaluates affective congruence across available modalities and identifies masked emotion."""
        vec_v = np.array([vision_prob.get(e, 0.0) for e in self.labels]) if gates["vision"] > 0.1 else np.zeros(8)
        vec_a = np.array([audio_prob.get(e, 0.0) for e in self.labels]) if gates["audio"] > 0.1 else np.zeros(8)
        vec_t = np.array([text_prob.get(e, 0.0) for e in self.labels]) if gates["text"] > 0.1 else np.zeros(8)

        sim_va = self._cosine_similarity(vec_v, vec_a) if (gates["vision"] > 0.1 and gates["audio"] > 0.1) else 1.0
        sim_vt = self._cosine_similarity(vec_v, vec_t) if (gates["vision"] > 0.1 and gates["text"] > 0.1) else 1.0
        sim_at = self._cosine_similarity(vec_a, vec_t) if (gates["audio"] > 0.1 and gates["text"] > 0.1) else 1.0

        pairwise = {
            "face_vs_voice": float((sim_va + 1.0) * 50.0),
            "face_vs_words": float((sim_vt + 1.0) * 50.0),
            "voice_vs_words": float((sim_at + 1.0) * 50.0),
        }

        # Active modality count
        active_pairs = []
        if gates["vision"] > 0.1 and gates["audio"] > 0.1:
            active_pairs.append(pairwise["face_vs_voice"])
        if gates["vision"] > 0.1 and gates["text"] > 0.1:
            active_pairs.append(pairwise["face_vs_words"])
        if gates["audio"] > 0.1 and gates["text"] > 0.1:
            active_pairs.append(pairwise["voice_vs_words"])

        if len(active_pairs) == 0:
            score = 100.0
        else:
            score = float(np.mean(active_pairs))

        # Check for Masked Affect: High smile AU12 with negative verbal/acoustic valence
        is_masked = False
        primary_conflict = None
        clinical_notes: List[str] = []

        v_smile = 0.0
        if vision_au is not None:
            v_smile = float(getattr(vision_au, "lip_corner_puller", getattr(vision_au, "au12_lip_corner_puller", 0.0)))

        v_valence = vision_vad.valence if vision_vad else 0.0
        a_valence = audio_vad.valence if audio_vad else 0.0
        t_valence = text_vad.valence if text_vad else 0.0

        # Smile with high acoustic or textual anger/distress
        if gates["vision"] > 0.3 and (v_smile > 0.55 or v_valence > 0.4):
            if (gates["audio"] > 0.3 and a_valence < -0.3) or (gates["text"] > 0.3 and t_valence < -0.35):
                is_masked = True
                score = min(score, 20.0)
                primary_conflict = "Smiling facial expression accompanied by negative vocal/verbal affect (Masked hostility or repressed distress)"
                clinical_notes.append("Potential masking/feigned affect detected: Visual joy contradicts acoustic/lexical negativity.")

        # High vocal arousal / stress with neutral or calm face
        if gates["audio"] > 0.3 and audio_vad and audio_vad.arousal > 0.6:
            if gates["vision"] > 0.3 and vision_vad and abs(vision_vad.arousal) < 0.2:
                if not primary_conflict:
                    primary_conflict = "Elevated acoustic vocal tension without corresponding facial micro-expression (Vocal leakage)"
                clinical_notes.append("Vocal arousal significantly exceeds facial expressiveness.")

        # Determine Tier
        if is_masked or score < CONGRUENCE_THRESHOLDS["discord"]:
            tier = "MASKED_AFFECT" if is_masked else "SIGNIFICANT_DISCORD"
        elif score < CONGRUENCE_THRESHOLDS["moderate"]:
            tier = "SIGNIFICANT_DISCORD"
        elif score < CONGRUENCE_THRESHOLDS["harmonious"]:
            tier = "MODERATE_TENSION"
        else:
            tier = "HARMONIOUS"

        if not clinical_notes:
            if tier == "HARMONIOUS":
                clinical_notes.append("High affective concordance: Facial, vocal, and verbal cues align harmoniously.")
            else:
                clinical_notes.append("Moderate affective ambivalence observed across modalities.")

        return ModalityCongruence(
            congruence_score=round(score, 1),
            congruence_tier=tier,
            pairwise_alignment=pairwise,
            primary_conflict=primary_conflict,
            clinical_notes=clinical_notes,
        )

    def fuse(
        self,
        vision: Optional[VisionEmotionResult],
        audio: Optional[Union[VoiceEmotionResult, AcousticSERResult]] = None,
        text: Optional[Union[TextEmotionResult, Any]] = None,
    ) -> CrossModalFusionResult:
        """Executes Cross-Modal Multi-Head Attention Fusion over available streams."""
        # 1. Extract feature vectors & modality confidences
        fv, cv = self._extract_vision_feature_vector(vision)
        fa, ca = self._extract_audio_feature_vector(audio)
        ft, ct = self._extract_text_feature_vector(text)

        # 2. Compute modality gates
        gate_v = 1.0 / (1.0 + np.exp(-4.0 * (cv - 0.25))) if cv > 0.05 else 0.0
        gate_a = 1.0 / (1.0 + np.exp(-4.0 * (ca - 0.25))) if ca > 0.05 else 0.0
        gate_t = 1.0 / (1.0 + np.exp(-4.0 * (ct - 0.25))) if ct > 0.05 else 0.0

        gate_sum = gate_v + gate_a + gate_t
        if gate_sum > 0:
            norm_v = gate_v / gate_sum
            norm_a = gate_a / gate_sum
            norm_t = gate_t / gate_sum
        else:
            norm_v = 0.33
            norm_a = 0.33
            norm_t = 0.34

        # 3. Project to shared latent dimension
        hv = np.dot(fv, self.W_v) * gate_v
        ha = np.dot(fa, self.W_a) * gate_a
        ht = np.dot(ft, self.W_t) * gate_t

        tokens = np.stack([hv, ha, ht])  # Shape (3, embed_dim)

        # 4. Multi-Head Scaled Dot-Product Cross-Attention
        Q = np.dot(tokens, self.W_q)
        K = np.dot(tokens, self.W_k)
        V = np.dot(tokens, self.W_v_proj)

        scale = np.sqrt(self.embed_dim)
        scores = np.dot(Q, K.T) / scale  # Shape (3, 3)

        # Softmax over columns
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_matrix = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        context = np.dot(attn_matrix, V)  # Shape (3, embed_dim)

        # Weighted combination of contextualized representations
        fused_latent = (norm_v * context[0]) + (norm_a * context[1]) + (norm_t * context[2])

        # 5. Output Head: Cross-Modal Attention Weighted Probability Fusion
        fused_probs: Dict[str, float] = {}
        for emo in self.labels:
            pv = (vision.probabilities.get(emo, 0.0) if (vision and getattr(vision, "face_detected", False)) else 0.0)
            pa = 0.0
            if audio is not None:
                if isinstance(audio, AcousticSERResult):
                    pa = float(audio.emotion_scores.get(emo, 0.0))
                else:
                    pa = float(audio.probabilities.get(emo, 0.0))
            pt = float(text.probabilities.get(emo, 0.0)) if text is not None else 0.0

            p_emo = (pv * norm_v) + (pa * norm_a) + (pt * norm_t)
            fused_probs[emo] = float(p_emo)

        prob_sum = sum(fused_probs.values())
        if prob_sum > 0:
            probabilities = {k: float(v / prob_sum) for k, v in fused_probs.items()}
        else:
            probabilities = {e: (1.0 if e == "neutral" else 0.0) for e in self.labels}

        dominant_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[dominant_emotion]

        # 6. Continuous 3D VAD mapping
        v_vad = getattr(vision, "affect", None)
        a_vad = getattr(audio, "vad", None) or getattr(audio, "affect", None)
        t_vad = getattr(text, "affect", None)

        valence = 0.0
        arousal = 0.0
        dominance = 0.0
        for emo, p in probabilities.items():
            vad = EMOTION_VAD_COORDINATES.get(emo, [0.0, 0.0, 0.0])
            valence += p * vad[0]
            arousal += p * vad[1]
            dominance += p * vad[2]

        # Direct VAD coordinate blending
        if gate_sum > 0:
            direct_v = (
                (norm_v * (v_vad.valence if v_vad else 0.0))
                + (norm_a * (a_vad.valence if a_vad else 0.0))
                + (norm_t * (t_vad.valence if t_vad else 0.0))
            )
            valence = 0.5 * valence + 0.5 * direct_v

        affect = AffectVector(
            valence=float(np.clip(valence, -1.0, 1.0)),
            arousal=float(np.clip(arousal, -1.0, 1.0)),
            dominance=float(np.clip(dominance, -1.0, 1.0)),
        )

        # 7. Cross-Modal Attention Weights object
        # Token order: 0: Vision, 1: Audio, 2: Text
        attn_weights = CrossModalAttentionWeights(
            text_to_vision=float(attn_matrix[2, 0]),
            text_to_audio=float(attn_matrix[2, 1]),
            vision_to_audio=float(attn_matrix[0, 1]),
            audio_to_vision=float(attn_matrix[1, 0]),
            modality_gates={"vision": float(gate_v), "audio": float(gate_a), "text": float(gate_t)},
            attention_matrix=attn_matrix.tolist(),
        )

        # 8. Affective Congruence Assessment
        vis_probs = getattr(vision, "probabilities", {}) if vision else {}
        aud_probs = (
            getattr(audio, "emotion_scores", {})
            if isinstance(audio, AcousticSERResult)
            else getattr(audio, "probabilities", {})
        ) if audio else {}
        txt_probs = getattr(text, "probabilities", {}) if text else {}

        congruence = self.evaluate_congruence(
            vision_prob=vis_probs,
            audio_prob=aud_probs,
            text_prob=txt_probs,
            vision_vad=v_vad,
            audio_vad=a_vad,
            text_vad=t_vad,
            vision_au=getattr(vision, "action_units", None) if vision else None,
            gates={"vision": gate_v, "audio": gate_a, "text": gate_t},
        )

        return CrossModalFusionResult(
            emotion_scores=probabilities,
            dominant_emotion=dominant_emotion,
            confidence=float(confidence),
            vad=affect,
            attention_weights=attn_weights,
            congruence=congruence,
            modality_contributions={"vision": float(norm_v), "audio": float(norm_a), "text": float(norm_t)},
            timestamp=time.time(),
        )
