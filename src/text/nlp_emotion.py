"""Advanced Real-Time NLP Affective Intelligence Engine.

Provides multi-dimensional text emotion recognition, emoji analysis, negation/intensifier
handling, token salience explainability, continuous 3D VAD projection, and empathy recommendations.
"""

import re
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from src.core.types import TextEmotionResult, AffectVector, TokenSalience, EmotionCategory


class TextEmotionClassifier:
    """Enterprise text emotion analyzer with multi-dimensional affective scoring."""

    # 3D VAD (Valence, Arousal, Dominance) Base Reference Coordinates per Emotion
    VAD_BASES: Dict[str, Tuple[float, float, float]] = {
        "joy": (0.81, 0.51, 0.46),
        "sadness": (-0.63, -0.27, -0.33),
        "anger": (-0.51, 0.59, 0.25),
        "fear": (-0.64, 0.60, -0.43),
        "surprise": (0.40, 0.67, -0.13),
        "disgust": (-0.60, 0.35, 0.11),
        "neutral": (0.00, 0.00, 0.00),
        "contempt": (-0.55, 0.38, 0.45),
        "love": (0.87, 0.54, 0.48),
        "excitement": (0.82, 0.75, 0.52),
        "optimism": (0.75, 0.42, 0.50),
        "anxiety": (-0.58, 0.65, -0.45),
        "frustration": (-0.65, 0.52, -0.10),
        "gratitude": (0.80, 0.35, 0.30),
        "confusion": (-0.15, 0.35, -0.30),
        "empathy": (0.68, 0.25, 0.20),
    }

    # Emotion Lexicon: Emotion category -> List of weighted words/stems
    EMOTION_LEXICON: Dict[str, Dict[str, float]] = {
        "joy": {
            "happy": 1.0, "happiness": 1.0, "glad": 0.85, "delighted": 1.0, "thrilled": 1.2,
            "ecstatic": 1.3, "joy": 1.1, "joyful": 1.1, "cheerful": 0.9, "wonderful": 1.0,
            "awesome": 0.95, "great": 0.8, "excellent": 0.95, "fantastic": 1.05, "amazing": 1.0,
            "love": 1.0, "loving": 0.9, "beloved": 0.95, "adore": 1.1, "blessed": 0.9,
            "celebrate": 1.0, "celebration": 1.0, "yay": 1.1, "hurray": 1.1, "win": 0.85,
            "winner": 0.9, "winning": 0.9, "smile": 0.8, "smiling": 0.85, "laugh": 0.9,
            "laughing": 0.9, "lol": 0.8, "lmao": 0.85, "haha": 0.85, "pleasant": 0.8,
            "brilliant": 0.9, "superb": 1.0, "splendid": 0.95, "triumph": 1.1, "perfect": 0.95,
            "glorious": 1.0, "proud": 0.85, "enjoy": 0.8, "enjoyed": 0.8, "fun": 0.8,
            "paradise": 0.95, "bliss": 1.2, "blissful": 1.2, "radiant": 0.9, "jubilant": 1.2,
            "pleased": 0.8, "content": 0.75, "satisfied": 0.8, "overjoyed": 1.3, "cheers": 0.8,
        },
        "sadness": {
            "sad": 1.0, "sadness": 1.0, "unhappy": 0.9, "sorrow": 1.1, "sorrowful": 1.1,
            "grief": 1.25, "grieving": 1.25, "cry": 1.0, "crying": 1.05, "tears": 0.9,
            "depressed": 1.2, "depression": 1.2, "gloomy": 0.85, "miserable": 1.15,
            "heartbroken": 1.3, "heartbreak": 1.25, "lonely": 1.0, "alone": 0.7,
            "hopeless": 1.2, "despair": 1.25, "down": 0.6, "hurt": 0.9, "pain": 0.85,
            "painful": 0.9, "mourn": 1.15, "mourning": 1.15, "regret": 0.85, "disappointed": 0.95,
            "disappointment": 0.95, "devastated": 1.3, "loss": 0.85, "lost": 0.7, "defeat": 0.85,
            "melancholy": 1.0, "unfortunate": 0.75, "tragic": 1.2, "tragedy": 1.2,
            "broken": 0.85, "weep": 1.05, "weeping": 1.05, "suffer": 0.95, "suffering": 1.0,
            "miss": 0.7, "missing": 0.75, "helpless": 1.05, "worthless": 1.15, "bleak": 0.9,
        },
        "anger": {
            "angry": 1.0, "anger": 1.0, "mad": 0.9, "furious": 1.3, "fury": 1.3,
            "rage": 1.35, "enraged": 1.35, "hate": 1.15, "hateful": 1.15, "irritated": 0.85,
            "irritating": 0.85, "annoyed": 0.85, "annoying": 0.85, "outraged": 1.25,
            "outrage": 1.25, "pissed": 1.1, "livid": 1.3, "infuriated": 1.3, "hostile": 1.1,
            "hostility": 1.1, "bitter": 0.9, "offended": 0.9, "scam": 1.0, "cheat": 0.95,
            "liar": 1.05, "bullshit": 1.2, "worst": 0.95, "terrible": 0.85, "horrible": 0.85,
            "threat": 0.95, "revenge": 1.1, "insult": 1.0, "insulting": 1.0, "fume": 1.1,
            "fuming": 1.15, "resent": 1.0, "resentment": 1.0, "wrath": 1.3, "sue": 0.9,
            "lawsuit": 0.85, "ridiculous": 0.8, "unacceptable": 1.0, "ruined": 0.85,
        },
        "fear": {
            "afraid": 1.0, "fear": 1.0, "fearful": 1.05, "scared": 1.0, "scary": 0.9,
            "terrified": 1.3, "terror": 1.3, "panic": 1.25, "panicked": 1.25, "anxious": 0.95,
            "anxiety": 1.0, "dread": 1.15, "dreading": 1.15, "alarmed": 0.9, "worried": 0.85,
            "worry": 0.85, "frightened": 1.1, "horrified": 1.25, "nervous": 0.85, "shaking": 0.8,
            "nightmare": 1.05, "danger": 0.95, "dangerous": 0.95, "unsafe": 0.9, "threatened": 1.05,
            "intimidated": 1.0, "phobic": 1.1, "phobia": 1.1, "trembling": 0.9, "petrified": 1.3,
            "paralyzed": 1.0, "insecure": 0.8, "uneasy": 0.75, "apprehensive": 0.9,
        },
        "surprise": {
            "surprised": 1.0, "surprise": 1.0, "shocked": 1.15, "shock": 1.1, "astonished": 1.2,
            "astonishing": 1.2, "amazed": 1.05, "amaze": 1.0, "staggered": 1.1, "unexpected": 0.95,
            "unbelievable": 1.0, "whoa": 1.1, "wow": 1.1, "omg": 1.15, "unreal": 0.9,
            "speechless": 1.1, "stunned": 1.2, "stunning": 0.95, "startled": 1.05, "abrupt": 0.75,
            "sudden": 0.75, "jaw-dropping": 1.25, "flabbergasted": 1.3, "mind-blowing": 1.25,
        },
        "disgust": {
            "disgusted": 1.1, "disgust": 1.1, "disgusting": 1.15, "gross": 1.05, "nasty": 1.0,
            "repulsive": 1.25, "repulsed": 1.2, "revolting": 1.25, "vile": 1.2, "sickening": 1.2,
            "sick": 0.75, "yuck": 1.1, "eww": 1.1, "foul": 1.05, "rotten": 0.95, "toxic": 0.9,
            "abhorrent": 1.3, "loathsome": 1.3, "distasteful": 0.9, "creepy": 0.85, "repugnant": 1.3,
        },
        "contempt": {
            "contempt": 1.2, "pathetic": 1.2, "worthless": 1.1, "useless": 1.0, "loser": 1.15,
            "ridiculous": 0.85, "arrogant": 0.95, "snob": 0.95, "despise": 1.2, "scorn": 1.2,
            "disdain": 1.15, "laughable": 0.9, "condescending": 1.05, "mockery": 1.0, "mock": 0.9,
            "inferior": 0.95, "hypocrite": 1.1, "clown": 0.8,
        },
        "neutral": {
            "okay": 0.7, "ok": 0.7, "fine": 0.65, "normal": 0.8, "average": 0.8,
            "standard": 0.8, "regular": 0.8, "neutral": 0.9, "moderate": 0.75,
            "inform": 0.7, "status": 0.7, "system": 0.6, "update": 0.6, "report": 0.6,
            "process": 0.6, "schedule": 0.6, "confirm": 0.65, "confirmed": 0.65, "yes": 0.5,
        }
    }

    # Extended Affective Nuances
    NUANCE_LEXICON: Dict[str, Dict[str, float]] = {
        "love": {"love": 1.2, "darling": 1.1, "sweetheart": 1.1, "babe": 0.9, "forever": 0.8, "romantic": 1.0, "adore": 1.1},
        "gratitude": {"thank": 1.1, "thanks": 1.0, "thankful": 1.2, "grateful": 1.2, "appreciate": 1.1, "appreciation": 1.1},
        "excitement": {"excited": 1.2, "pumped": 1.1, "hyped": 1.1, "cannot wait": 1.2, "thrilled": 1.2, "fired up": 1.2},
        "optimism": {"hope": 1.0, "hopeful": 1.1, "optimistic": 1.2, "looking forward": 1.1, "positive": 0.9, "bright future": 1.2},
        "anxiety": {"anxious": 1.2, "stressed": 1.1, "overwhelmed": 1.2, "nervous": 1.0, "on edge": 1.1, "panic": 1.2},
        "frustration": {"frustrated": 1.2, "annoyed": 1.0, "stuck": 0.9, "waste of time": 1.2, "fed up": 1.2, "exhausting": 1.0},
        "confusion": {"confused": 1.2, "unclear": 1.0, "what": 0.6, "how come": 0.8, "puzzled": 1.1, "baffled": 1.2, "lost": 0.8},
        "empathy": {"understand": 0.9, "hear you": 1.0, "feel for you": 1.2, "sorry to hear": 1.2, "support you": 1.1, "here for you": 1.2},
    }

    # Emoji Affect Mapping (Emoji -> (Primary Emotion, Weight, Valence, Arousal))
    EMOJI_MAP: Dict[str, Tuple[str, float, float, float]] = {
        "😀": ("joy", 0.9, 0.8, 0.5), "😃": ("joy", 0.95, 0.85, 0.6), "😄": ("joy", 1.0, 0.9, 0.6),
        "😁": ("joy", 1.0, 0.85, 0.6), "😆": ("joy", 1.1, 0.9, 0.7), "😂": ("joy", 1.2, 0.95, 0.8),
        "🤣": ("joy", 1.3, 0.95, 0.85), "😊": ("joy", 0.85, 0.8, 0.4), "🥰": ("joy", 1.1, 0.9, 0.5),
        "😍": ("joy", 1.2, 0.95, 0.6), "🤩": ("surprise", 1.1, 0.85, 0.8), "😘": ("joy", 0.9, 0.8, 0.4),
        "🥳": ("joy", 1.2, 0.95, 0.8), "🎉": ("joy", 1.1, 0.9, 0.7), "✨": ("joy", 0.8, 0.7, 0.5),
        "❤️": ("joy", 1.1, 0.9, 0.5), "💖": ("joy", 1.1, 0.9, 0.5), "🔥": ("joy", 0.9, 0.75, 0.8),
        "👍": ("joy", 0.7, 0.6, 0.2), "🙌": ("joy", 1.0, 0.85, 0.6), "💯": ("joy", 0.9, 0.8, 0.5),
        "😢": ("sadness", 1.0, -0.7, -0.2), "😭": ("sadness", 1.25, -0.85, 0.4), "🥺": ("sadness", 0.9, -0.5, -0.1),
        "😞": ("sadness", 0.9, -0.65, -0.3), "😔": ("sadness", 0.95, -0.7, -0.3), "💔": ("sadness", 1.2, -0.9, -0.2),
        "☹️": ("sadness", 0.85, -0.6, -0.2), "🙁": ("sadness", 0.8, -0.55, -0.2), "😿": ("sadness", 0.9, -0.7, -0.2),
        "😡": ("anger", 1.2, -0.8, 0.8), "😠": ("anger", 1.0, -0.7, 0.7), "🤬": ("anger", 1.4, -0.9, 0.9),
        "👿": ("anger", 1.1, -0.75, 0.7), "😤": ("anger", 0.9, -0.5, 0.6), "💢": ("anger", 1.0, -0.6, 0.7),
        "😨": ("fear", 1.1, -0.7, 0.7), "😰": ("fear", 1.0, -0.65, 0.6), "😱": ("fear", 1.3, -0.75, 0.9),
        "😳": ("surprise", 0.9, -0.1, 0.6), "🤯": ("surprise", 1.2, 0.2, 0.9), "😲": ("surprise", 1.0, 0.1, 0.7),
        "😮": ("surprise", 0.85, 0.1, 0.6), "😯": ("surprise", 0.8, 0.1, 0.5),
        "🤢": ("disgust", 1.15, -0.8, 0.4), "🤮": ("disgust", 1.35, -0.9, 0.6), "💩": ("disgust", 0.8, -0.5, 0.2),
        "😒": ("contempt", 0.95, -0.5, 0.2), "🙄": ("contempt", 1.0, -0.4, 0.3), "😏": ("contempt", 0.8, 0.2, 0.3),
        "😐": ("neutral", 0.7, 0.0, 0.0), "😑": ("neutral", 0.75, -0.1, 0.0), "🤔": ("neutral", 0.65, 0.1, 0.2),
    }

    # Negation words that invert or diminish sentiment of upcoming words
    NEGATION_WORDS = {
        "not", "never", "no", "hardly", "scarcely", "barely", "wasn't", "weren't",
        "isn't", "aren't", "don't", "doesn't", "didn't", "can't", "cannot", "won't",
        "wouldn't", "couldn't", "shouldn't", "haven't", "hasn't", "hadn't", "without",
        "neither", "nor", "none", "nobody", "nowhere"
    }

    # Intensifiers and diminishers
    INTENSIFIERS = {
        "extremely": 1.5, "incredibly": 1.5, "very": 1.3, "so": 1.25, "really": 1.25,
        "super": 1.3, "totally": 1.3, "completely": 1.35, "absolutely": 1.4, "immensely": 1.4,
        "deeply": 1.3, "hugely": 1.3, "overwhelmingly": 1.4, "tremendously": 1.4, "insanely": 1.45,
        "unbelievably": 1.4, "utterly": 1.4, "fucking": 1.4, "hell": 1.25, "damned": 1.25
    }

    DIMINISHERS = {
        "slightly": 0.6, "barely": 0.5, "hardly": 0.5, "a bit": 0.7, "a little": 0.7,
        "somewhat": 0.75, "kind of": 0.75, "kinda": 0.75, "sort of": 0.75, "moderately": 0.8,
        "partly": 0.7, "marginally": 0.6
    }

    def __init__(self):
        """Initializes the NLP emotion analyzer."""
        pass

    def analyze_text(self, text: str) -> TextEmotionResult:
        """Analyzes single text message and returns full affective profile."""
        if not text or not text.strip():
            return TextEmotionResult(
                text="",
                dominant_emotion="neutral",
                confidence=1.0,
                probabilities={"neutral": 1.0, "joy": 0.0, "sadness": 0.0, "anger": 0.0,
                               "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "contempt": 0.0},
                affect=AffectVector(0.0, 0.0, 0.0),
                empathy_advice="No text content provided."
            )

        clean_text = text.strip()
        tokens = self._tokenize(clean_text)
        emojis = self._extract_emojis(clean_text)

        # Initialize raw emotion score accumulators
        raw_scores: Dict[str, float] = {
            "joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0,
            "surprise": 0.0, "disgust": 0.0, "neutral": 0.0, "contempt": 0.0
        }
        nuance_scores: Dict[str, float] = {k: 0.0 for k in self.NUANCE_LEXICON}
        salience_tokens: List[TokenSalience] = []

        total_words = len(tokens)
        exclamation_count = clean_text.count("!")
        question_count = clean_text.count("?")
        is_all_caps = clean_text.isupper() and len(clean_text) > 3

        # Window-based token analysis with Negation and Intensifier lookahead
        negation_active = False
        negation_countdown = 0
        current_multiplier = 1.0

        for i, token in enumerate(tokens):
            token_lower = token.lower()

            # Check negation
            if token_lower in self.NEGATION_WORDS:
                negation_active = True
                negation_countdown = 3  # applies to next 3 words or until punctuation
                salience_tokens.append(TokenSalience(token, 0.2, "negation", 0.0, is_negated=True))
                continue

            # Check intensifier
            if token_lower in self.INTENSIFIERS:
                current_multiplier = self.INTENSIFIERS[token_lower]
                salience_tokens.append(TokenSalience(token, 0.3, "intensifier", 0.0, is_intensified=True))
                continue

            # Check diminisher
            if token_lower in self.DIMINISHERS:
                current_multiplier = self.DIMINISHERS[token_lower]
                salience_tokens.append(TokenSalience(token, 0.2, "diminisher", 0.0))
                continue

            # Match against 8 Core Emotions
            matched_emotion = None
            token_score = 0.0

            for emo_name, word_map in self.EMOTION_LEXICON.items():
                # Direct match or prefix match
                if token_lower in word_map:
                    base_val = word_map[token_lower]
                    token_score = base_val * current_multiplier
                    matched_emotion = emo_name
                    break
                else:
                    # check simple stem
                    for root_w, base_val in word_map.items():
                        if len(root_w) >= 4 and token_lower.startswith(root_w):
                            token_score = base_val * 0.85 * current_multiplier
                            matched_emotion = emo_name
                            break
                if matched_emotion:
                    break

            if matched_emotion:
                # Handle Negation: flip or attenuate
                if negation_active:
                    if matched_emotion == "joy":
                        raw_scores["sadness"] += token_score * 0.85
                        salience_tokens.append(TokenSalience(token, token_score, "sadness (negated joy)", -0.6, is_negated=True))
                    elif matched_emotion in ("sadness", "anger", "fear", "disgust"):
                        raw_scores["neutral"] += token_score * 0.6
                        raw_scores["joy"] += token_score * 0.4
                        salience_tokens.append(TokenSalience(token, token_score, "relief (negated neg)", 0.5, is_negated=True))
                    else:
                        raw_scores["neutral"] += token_score * 0.5
                        salience_tokens.append(TokenSalience(token, token_score, "neutral (negated)", 0.0, is_negated=True))
                else:
                    raw_scores[matched_emotion] += token_score
                    pol = 0.7 if matched_emotion in ("joy", "surprise") else (-0.7 if matched_emotion != "neutral" else 0.0)
                    salience_tokens.append(TokenSalience(token, token_score, matched_emotion, pol, is_intensified=current_multiplier > 1.0))
            else:
                # Neutral baseline token
                salience_tokens.append(TokenSalience(token, 0.05, "neutral", 0.0))

            # Match nuanced emotions
            for nuance_name, n_words in self.NUANCE_LEXICON.items():
                for nw, nw_val in n_words.items():
                    if nw in clean_text.lower() or token_lower == nw:
                        nuance_scores[nuance_name] += nw_val * (0.3 if negation_active else 1.0)

            # Decrement negation countdown
            if negation_active:
                negation_countdown -= 1
                if negation_countdown <= 0:
                    negation_active = False

            # Reset multiplier for subsequent tokens
            current_multiplier = 1.0

        # Incorporate Emoji Affect
        for emoji_char in emojis:
            if emoji_char in self.EMOJI_MAP:
                emo_cat, weight, val, ar = self.EMOJI_MAP[emoji_char]
                raw_scores[emo_cat] += weight * 1.5
                salience_tokens.append(TokenSalience(emoji_char, weight * 1.5, emo_cat, val, is_intensified=True))

        # Punctuation & Caps Boost
        if exclamation_count > 0:
            boost = min(exclamation_count * 0.25, 0.75)
            # Arousal boost for leading emotions
            for e in ("joy", "anger", "surprise", "fear"):
                if raw_scores[e] > 0.1:
                    raw_scores[e] += boost

        if is_all_caps and total_words > 1:
            if raw_scores["anger"] > 0:
                raw_scores["anger"] *= 1.35
            if raw_scores["joy"] > 0:
                raw_scores["joy"] *= 1.25

        # If no significant emotional tokens were found, default to neutral
        total_emotion_mass = sum(raw_scores.values())
        if total_emotion_mass < 0.15:
            raw_scores["neutral"] = max(1.2, 0.3 * total_words)

        # Softmax / Normalized probability distribution
        probabilities = self._compute_probabilities(raw_scores)

        # Dominant Emotion & Confidence
        dominant_emotion = max(probabilities.items(), key=lambda x: x[1])[0]
        confidence = probabilities[dominant_emotion]

        # Continuous 3D VAD Vector Projection
        affect = self._compute_vad_vector(probabilities, exclamation_count, is_all_caps)

        # Compute Polarity and Subjectivity
        pos_mass = probabilities.get("joy", 0.0) + 0.3 * probabilities.get("surprise", 0.0)
        neg_mass = (probabilities.get("sadness", 0.0) + probabilities.get("anger", 0.0) +
                    probabilities.get("fear", 0.0) + probabilities.get("disgust", 0.0) +
                    probabilities.get("contempt", 0.0))
        polarity = max(-1.0, min(1.0, (pos_mass - neg_mass) * 1.2))
        subjectivity = max(0.0, min(1.0, 1.0 - probabilities.get("neutral", 0.0)))

        # Normalize Nuanced Emotions
        top_nuances = {k: round(v / (sum(nuance_scores.values()) + 1e-5), 3)
                       for k, v in nuance_scores.items() if v > 0.2}

        # Empathy Advice Generation
        empathy_advice = self._generate_empathy_advice(dominant_emotion, confidence, affect)

        return TextEmotionResult(
            text=clean_text,
            dominant_emotion=dominant_emotion,
            confidence=round(confidence, 3),
            probabilities={k: round(v, 4) for k, v in probabilities.items()},
            nuanced_emotions=top_nuances,
            affect=affect,
            polarity=round(polarity, 3),
            subjectivity=round(subjectivity, 3),
            salience_tokens=salience_tokens,
            empathy_advice=empathy_advice,
            emojis_detected=emojis,
            timestamp=time.time(),
        )

    def _tokenize(self, text: str) -> List[str]:
        """Splits text into words, preserving contractions and punctuation tokens."""
        clean = re.sub(r"[^\w\s\'-]", " ", text)
        return [w.strip() for w in clean.split() if w.strip()]

    def _extract_emojis(self, text: str) -> List[str]:
        """Extracts emoji characters from text."""
        return [c for c in text if c in self.EMOJI_MAP]

    def _compute_probabilities(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Applies calibrated Softmax temperature over raw scores."""
        temperature = 0.85
        exp_scores = {k: math.exp(v / temperature) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        return {k: exp_scores[k] / total_exp for k in scores}

    def _compute_vad_vector(self, probabilities: Dict[str, float], exclamations: int, is_caps: bool) -> AffectVector:
        """Projects discrete emotion probability distribution into 3D continuous VAD space."""
        v = sum(probabilities.get(e, 0.0) * self.VAD_BASES[e][0] for e in probabilities if e in self.VAD_BASES)
        a = sum(probabilities.get(e, 0.0) * self.VAD_BASES[e][1] for e in probabilities if e in self.VAD_BASES)
        d = sum(probabilities.get(e, 0.0) * self.VAD_BASES[e][2] for e in probabilities if e in self.VAD_BASES)

        # Exclamation & caps arousal boost
        if exclamations > 0:
            a += min(exclamations * 0.08, 0.25)
        if is_caps:
            a += 0.15

        # Clamp between -1.0 and +1.0
        v = max(-1.0, min(1.0, v))
        a = max(-1.0, min(1.0, a))
        d = max(-1.0, min(1.0, d))

        return AffectVector(valence=round(v, 3), arousal=round(arousal_val := a, 3), dominance=round(d, 3))

    def _generate_empathy_advice(self, dominant_emotion: str, confidence: float, affect: AffectVector) -> str:
        """Generates actionable empathy and communication response suggestions."""
        if dominant_emotion == "joy":
            return "🎉 Match positive energy! Validate their excitement and celebrate achievements with enthusiastic, warm affirmations."
        elif dominant_emotion == "sadness":
            return "💙 Express genuine warmth and supportive presence. Avoid toxic positivity; offer listening, patience, and validation."
        elif dominant_emotion == "anger":
            return "🛡️ De-escalate immediately. Acknowledge frustration calmly, validate legitimate pain points, and focus on clear, concrete solutions without defensiveness."
        elif dominant_emotion == "fear":
            return "🤝 Provide reassurance, structure, and psychological safety. Clarify next steps and remove uncertainty step-by-step."
        elif dominant_emotion == "surprise":
            return "💡 Clarify context and anchor the new information. Check if the surprise is positive (celebrate) or disorienting (provide clarity)."
        elif dominant_emotion == "disgust":
            return "🧼 Acknowledge boundaries and distaste. Reframe or offer clean alternatives to resolve the friction point."
        elif dominant_emotion == "contempt":
            return "⚖️ Maintain calm professionalism and strong mutual respect. Focus on objective facts and bridge misaligned expectations."
        else:
            return "✨ Clear, constructive, and balanced communication is recommended. Keep engagement crisp and responsive."
