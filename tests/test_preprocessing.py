"""
Unit Tests for Text Preprocessing and Input Validation
"""

import unittest
from src.preprocessing import clean_text, validate_input, extract_key_terms


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_html(self):
        raw = "<p>Breaking News &amp; Alert: <b>Scientists</b> discover water!</p>"
        cleaned = clean_text(raw)
        self.assertIn("Breaking News & Alert", cleaned)
        self.assertNotIn("&amp;", cleaned)

    def test_clean_text_whitespace(self):
        raw = "Headline    with   tabs\t\tand\n\nnewlines"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Headline with tabs and newlines")

    def test_validate_input_valid(self):
        valid, err = validate_input("NASA Rover Finds Ancient River Delta on Mars", "Scientists confirmed that sedimentary layers indicate persistent water flow in the crater billions of years ago.")
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_validate_input_empty_headline(self):
        valid, err = validate_input("", "Some article body with enough words to analyze.")
        self.assertFalse(valid)
        self.assertIn("Headline is required", err)

    def test_validate_input_short_headline(self):
        valid, err = validate_input("Hello there", "Some body text here with sufficient content.")
        self.assertFalse(valid)
        self.assertIn("Headline is too short", err)

    def test_extract_key_terms(self):
        headline = "President Biden Announces New Federal Infrastructure Investments in Chicago"
        terms = extract_key_terms(headline)
        self.assertIn("President", terms)
        self.assertIn("Biden", terms)
        self.assertIn("Chicago", terms)
        self.assertNotIn("in", terms)


if __name__ == "__main__":
    unittest.main()
