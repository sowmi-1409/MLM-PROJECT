"""
Fake News Classifier Module
Loads trained model pipeline, performs inference, generates confidence scores,
and computes local feature attributions (explainability).
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List
from src.preprocessing import clean_text

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fake_news_model.joblib")


class FakeNewsClassifier:
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
                self.model_name = data.get("model_name", "Linear Model")
            except Exception as e:
                print(f"[FakeNewsClassifier] Error loading model: {e}")

    @property
    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.classifier is not None

    def predict(self, headline: str, body: str) -> Dict[str, Any]:
        """
        Classify news article text as Real/Credible or Fake/Unreliable.
        Returns prediction, confidence, probability breakdown, and influential words.
        """
        if not self.is_ready:
            return {
                "status": "unavailable",
                "label": "Evaluation Unavailable",
                "confidence": 0.0,
                "probabilities": {"REAL": 0.5, "FAKE": 0.5},
                "influential_features": [],
                "explanation": "Model weights have not been trained or loaded yet."
            }

        full_text = clean_text(headline)
        if body:
            full_text += " — " + clean_text(body)

        vec = self.vectorizer.transform([full_text])

        # Probabilities
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(vec)[0]
            prob_real = float(probs[0])
            prob_fake = float(probs[1])
        elif hasattr(self.classifier, "decision_function"):
            decision = float(self.classifier.decision_function(vec)[0])
            prob_fake = 1.0 / (1.0 + np.exp(-decision))
            prob_real = 1.0 - prob_fake
        else:
            pred = int(self.classifier.predict(vec)[0])
            prob_fake = 0.9 if pred == 1 else 0.1
            prob_real = 1.0 - prob_fake

        pred_class = 1 if prob_fake >= 0.5 else 0
        label_name = "Unreliable / Fake" if pred_class == 1 else "Credible / Real"
        confidence = prob_fake if pred_class == 1 else prob_real

        # Feature Attribution (Explainability)
        influential_words = self._extract_influential_features(vec, full_text, top_n=6)

        explanation = (
            f"The model classifies this text as {label_name} with {confidence * 100:.1f}% confidence "
            f"based on vocabulary patterns, stylistic markers, and discourse structures learned from the benchmark training corpus."
        )

        return {
            "status": "success",
            "prediction_code": pred_class,
            "label": label_name,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 1),
            "probabilities": {
                "real": round(prob_real, 4),
                "fake": round(prob_fake, 4)
            },
            "model_name": self.model_name,
            "influential_features": influential_words,
            "explanation": explanation,
            "limitations": (
                "Statistical text classification identifies linguistic correlations and lexical cues. "
                "It does not verify external factual truth in the physical world; always corroborate breaking news with primary wire services."
            )
        }

    def _extract_influential_features(self, vec, text: str, top_n: int = 6) -> List[Dict[str, Any]]:
        """Extract top positive and negative local word contributions."""
        if not hasattr(self.classifier, "coef_"):
            return []

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        coefs = self.classifier.coef_[0]

        # Nonzero features in this specific text
        nonzero_indices = vec.nonzero()[1]
        if len(nonzero_indices) == 0:
            return []

        contributions = []
        for idx in nonzero_indices:
            term = feature_names[idx]
            weight = vec[0, idx] * coefs[idx]
            contributions.append({
                "word": term,
                "weight": round(float(weight), 3),
                "lean": "Fake" if weight > 0 else "Real"
            })

        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x["weight"]), reverse=True)
        return contributions[:top_n]
