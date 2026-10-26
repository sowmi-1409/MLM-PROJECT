"""
Clickbait Detection Module
Loads trained model pipeline, performs inference, generates confidence scores,
identifies clickbait stylistic indicators, and computes local feature attributions.
"""

import os
import re
import joblib
import numpy as np
from typing import Dict, Any, List
from src.preprocessing import clean_text

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "clickbait_model.joblib")

# Common linguistic patterns found in sensational headlines
CLICKBAIT_PATTERNS = [
    (r"\b(\d+)\s+(ways|reasons|things|secrets|tricks|signs|times|facts)\b", "Listicle formatting"),
    (r"\b(will blow your mind|you won't believe|will make you cry|shocks everyone|jaw-dropping)\b", "Sensational exaggeration"),
    (r"\b(this is why|here's what|here is what|the real reason why)\b", "Curiosity-gap forward referencing"),
    (r"\b(what happens next|see what happened)\b", "Unresolved suspense lure"),
    (r"\b(unbelievable|astonishing|insane|mind-blowing|shocking)\b", "Intensifier hyperbole"),
    (r"[!?]{2,}", "Exclamatory punctuation clustering")
]


class ClickbaitDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.vectorizer = None
        self.classifier = None
        self.model_name = "Unknown"
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.vectorizer = data.get("vectorizer")
                self.classifier = data.get("classifier")
                self.model_name = data.get("model_name", "MultinomialNB")
            except Exception as e:
                print(f"[ClickbaitDetector] Error loading model: {e}")

    @property
    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.classifier is not None

    def predict(self, headline: str) -> Dict[str, Any]:
        """
        Classify a headline as Clickbait or Non-Clickbait.
        """
        clean_h = clean_text(headline)
        if not self.is_ready or not clean_h:
            return {
                "status": "unavailable",
                "label": "Evaluation Unavailable",
                "confidence": 0.0,
                "probabilities": {"non_clickbait": 0.5, "clickbait": 0.5},
                "stylistic_cues": [],
                "influential_words": [],
                "explanation": "Clickbait detection pipeline is not loaded."
            }

        vec = self.vectorizer.transform([clean_h])

        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(vec)[0]
            prob_non = float(probs[0])
            prob_cb = float(probs[1])
        elif hasattr(self.classifier, "decision_function"):
            decision = float(self.classifier.decision_function(vec)[0])
            prob_cb = 1.0 / (1.0 + np.exp(-decision))
            prob_non = 1.0 - prob_cb
        else:
            pred = int(self.classifier.predict(vec)[0])
            prob_cb = 0.9 if pred == 1 else 0.1
            prob_non = 1.0 - prob_cb

        # Scan for explicit rhetorical clickbait patterns
        stylistic_cues = []
        for pattern, description in CLICKBAIT_PATTERNS:
            match = re.search(pattern, clean_h, re.IGNORECASE)
            if match:
                stylistic_cues.append({
                    "pattern": description,
                    "matched_text": match.group(0)
                })

        pred_class = 1 if prob_cb >= 0.5 else 0
        label_name = "Clickbait Headline" if pred_class == 1 else "Non-Clickbait (Informative)"
        confidence = prob_cb if pred_class == 1 else prob_non

        # Local word attribution
        influential_words = self._extract_influential_features(vec, clean_h)

        if pred_class == 1:
            explanation = (
                f"Flagged as Clickbait with {confidence * 100:.1f}% confidence. "
                f"Headline exhibits high sensory arousal, curiosity-gap phrasing, or listicle structure designed to provoke clicks."
            )
        else:
            explanation = (
                f"Classified as Non-Clickbait with {confidence * 100:.1f}% confidence. "
                f"Headline follows standard informative, journalistic phrasing that reveals the core news event directly."
            )

        return {
            "status": "success",
            "prediction_code": pred_class,
            "label": label_name,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 1),
            "probabilities": {
                "non_clickbait": round(prob_non, 4),
                "clickbait": round(prob_cb, 4)
            },
            "model_name": self.model_name,
            "stylistic_cues": stylistic_cues,
            "influential_words": influential_words,
            "explanation": explanation,
            "limitations": (
                "Clickbait detection assesses headline presentation style, not factual validity. "
                "Legitimate journalism may occasionally use enticing headlines, while deceptive articles may use calm headlines."
            )
        }

    def _extract_influential_features(self, vec, text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Extract top influential words from Naive Bayes log probability ratios or linear weights."""
        if not hasattr(self.classifier, "feature_log_prob_"):
            return []

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        # Log ratio: log(P(w | Clickbait)) - log(P(w | Non-Clickbait))
        log_ratio = self.classifier.feature_log_prob_[1] - self.classifier.feature_log_prob_[0]

        nonzero_indices = vec.nonzero()[1]
        if len(nonzero_indices) == 0:
            return []

        contributions = []
        for idx in nonzero_indices:
            term = feature_names[idx]
            impact = log_ratio[idx]
            contributions.append({
                "word": term,
                "weight": round(float(impact), 3),
                "lean": "Clickbait" if impact > 0 else "Informative"
            })

        contributions.sort(key=lambda x: abs(x["weight"]), reverse=True)
        return contributions[:top_n]
