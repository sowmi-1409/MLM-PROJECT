"""
Headline-Body Mismatch Detection Module
Measures lexical and semantic alignment between article headlines and body content
using calibrated TF-IDF Cosine Similarity, entity grounding, and Jaccard overlap.
"""

import os
import re
import joblib
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import clean_text, extract_key_terms

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "mismatch_detector.joblib")


class MismatchDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.vectorizer = None
        self.threshold = 0.100  # Empirically calibrated default
        self.metadata = {}
        self._load_pipeline()

    def _load_pipeline(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.vectorizer = data.get("vectorizer")
                self.threshold = float(data.get("threshold", 0.100))
                self.metadata = data.get("metadata", {})
            except Exception as e:
                print(f"[MismatchDetector] Error loading model: {e}")

    @property
    def is_ready(self) -> bool:
        return self.vectorizer is not None

    def analyze(self, headline: str, body: str) -> Dict[str, Any]:
        """
        Evaluate semantic and lexical consistency between headline and article body.
        """
        clean_h = clean_text(headline)
        clean_b = clean_text(body)

        if not clean_b:
            return {
                "status": "partial",
                "label": "Body Text Not Provided",
                "similarity_score": 0.0,
                "threshold": self.threshold,
                "shared_keywords": [],
                "unmatched_headline_terms": [],
                "explanation": "No article body was supplied to compare against the headline.",
                "limitations": "Consistency analysis requires both headline and body text."
            }

        # 1. Cosine similarity using fitted vectorizer
        if self.is_ready:
            vecs = self.vectorizer.transform([clean_h, clean_b])
            cos_sim = float(cosine_similarity(vecs[0], vecs[1])[0][0])
        else:
            # Fallback simple bag-of-words cosine similarity
            cos_sim = self._fallback_cosine(clean_h, clean_b)

        # 2. Key entity / keyword grounding
        headline_terms = extract_key_terms(clean_h, max_terms=8)
        body_lower = clean_b.lower()

        shared_terms = [t for t in headline_terms if t.lower() in body_lower]
        missing_terms = [t for t in headline_terms if t.lower() not in body_lower]

        overlap_ratio = len(shared_terms) / max(len(headline_terms), 1)

        # 3. Categorization based on calibrated threshold
        # Threshold was tuned on real news vs shuffled news (validation optimal ~0.100)
        high_thresh = self.threshold * 1.8
        moderate_thresh = self.threshold

        if cos_sim >= high_thresh:
            status_code = "consistent"
            label = "Consistent & Well-Aligned"
            badge = "success"
            explanation = (
                f"Headline is strongly corroborated by the article body (Cosine Similarity: {cos_sim:.3f}, above threshold {self.threshold:.3f}). "
                f"Key themes and entities appear prominently in both sections."
            )
        elif cos_sim >= moderate_thresh:
            status_code = "moderate"
            label = "Moderate Similarity (Acceptable)"
            badge = "info"
            explanation = (
                f"Headline shows moderate alignment with the article body (Cosine Similarity: {cos_sim:.3f}). "
                f"Core topic is present, though headline may summarize only a specific facet or lead with an angle."
            )
        else:
            status_code = "mismatch"
            label = "Possible Headline–Body Mismatch"
            badge = "warning"
            explanation = (
                f"Headline exhibits low semantic overlap with the article body (Cosine Similarity: {cos_sim:.3f}, below calibrated threshold {self.threshold:.3f}). "
                f"This divergence may indicate clickbait bait-and-switch framing, sensational hyperbole, or an unrelated headline."
            )

        return {
            "status": "success",
            "status_code": status_code,
            "badge": badge,
            "label": label,
            "similarity_score": round(cos_sim, 4),
            "similarity_percentage": round(min(cos_sim * 250, 100), 1),  # Scaled for intuitive visual gauge
            "calibrated_threshold": round(self.threshold, 3),
            "shared_keywords": shared_terms,
            "unmatched_headline_terms": missing_terms,
            "overlap_ratio": round(overlap_ratio, 2),
            "explanation": explanation,
            "limitations": (
                "Important: Low lexical similarity indicates linguistic divergence or possible sensational framing; "
                "it does not automatically prove factual falsity. An article may truthfully report breaking developments using metaphor or novel terminology."
            )
        }

    def _fallback_cosine(self, h: str, b: str) -> float:
        """Fallback bag-of-words cosine similarity if vectorizer not loaded."""
        words_h = set(re.findall(r"\w+", h.lower()))
        words_b = set(re.findall(r"\w+", b.lower()))
        if not words_h or not words_b:
            return 0.0
        intersection = words_h.intersection(words_b)
        return float(len(intersection) / (np.sqrt(len(words_h)) * np.sqrt(len(words_b))))
