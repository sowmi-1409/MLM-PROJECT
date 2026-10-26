"""
Text Preprocessing and Input Validation Utilities for NewsGuard
"""

import re
import html
import unicodedata
from typing import Tuple, List, Optional


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text:
    - Decode HTML entities
    - Normalize Unicode characters (NFKD)
    - Strip excessive whitespace, tabs, and newlines
    - Retain punctuation relevant for sentiment/emphasis
    """
    if not text or not isinstance(text, str):
        return ""

    # Decode HTML
    text = html.unescape(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Standardize whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def validate_input(headline: str, body: str) -> Tuple[bool, Optional[str]]:
    """
    Validate analysis inputs.
    Returns (is_valid, error_message).
    """
    clean_h = clean_text(headline)
    clean_b = clean_text(body)

    if not clean_h:
        return False, "Headline is required. Please provide a news headline to analyze."

    if len(clean_h.split()) < 3:
        return False, "Headline is too short. Please provide at least 3 words in the headline."

    if len(clean_h) > 500:
        return False, "Headline exceeds maximum permitted length (500 characters)."

    if clean_b and len(clean_b.split()) < 10:
        return False, "Article body is too short. Please provide at least 10 words for article body analysis."

    return True, None


def extract_key_terms(text: str, max_terms: int = 8) -> List[str]:
    """
    Extract key salient terms (capitalized entities and longer non-stopwords)
    useful for external search and lexical overlap analysis.
    """
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "this", "that",
        "these", "those", "it", "its", "they", "them", "their", "we", "us", "our", "you",
        "your", "he", "him", "his", "she", "her", "said", "says", "new", "will", "can"
    }

    words = re.findall(r"\b[A-Za-z0-9\-\']+\b", text)
    filtered = [w for w in words if w.lower() not in STOPWORDS and len(w) > 3]

    # Prioritize capitalized terms / proper nouns
    proper_nouns = [w for w in filtered if w[0].isupper()]
    other_words = [w for w in filtered if not w[0].isupper()]

    combined = proper_nouns + other_words
    seen = set()
    result = []
    for term in combined:
        t_low = term.lower()
        if t_low not in seen:
            seen.add(t_low)
            result.append(term)
        if len(result) >= max_terms:
            break

    return result
