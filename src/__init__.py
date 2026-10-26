"""
NewsGuard AI Engine Package
"""

from src.preprocessing import clean_text, validate_input, extract_key_terms
from src.fake_news_classifier import FakeNewsClassifier
from src.clickbait_detector import ClickbaitDetector
from src.mismatch_detector import MismatchDetector
from src.external_verifier import ExternalVerifier
from src.explainability import ExplainabilityEngine

__all__ = [
    "clean_text",
    "validate_input",
    "extract_key_terms",
    "FakeNewsClassifier",
    "ClickbaitDetector",
    "MismatchDetector",
    "ExternalVerifier",
    "ExplainabilityEngine"
]
