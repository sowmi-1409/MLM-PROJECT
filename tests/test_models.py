"""
Unit Tests for Machine Learning Inference Modules
"""

import unittest
from src.fake_news_classifier import FakeNewsClassifier
from src.clickbait_detector import ClickbaitDetector
from src.mismatch_detector import MismatchDetector
from src.external_verifier import ExternalVerifier
from src.explainability import ExplainabilityEngine


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake_clf = FakeNewsClassifier()
        cls.cb_detector = ClickbaitDetector()
        cls.mm_detector = MismatchDetector()
        cls.verifier = ExternalVerifier()

    def test_fake_news_classifier_readiness(self):
        self.assertTrue(self.fake_clf.is_ready)

    def test_fake_news_inference_output(self):
        headline = "Department of Justice Releases Annual Crime Statistics Report"
        body = "The United States Department of Justice published its comprehensive annual report detailing law enforcement metrics and federal prosecutions across all fifty states."
        res = self.fake_clf.predict(headline, body)

        self.assertEqual(res["status"], "success")
        self.assertIn("prediction_code", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)
        self.assertIn("influential_features", res)
        self.assertIn(res["label"], ["Credible / Real", "Unreliable / Fake"])

    def test_clickbait_detector_readiness(self):
        self.assertTrue(self.cb_detector.is_ready)

    def test_clickbait_detection_clickbait_pattern(self):
        headline = "You Won't Believe What These 7 Celebrities Look Like Today!!"
        res = self.cb_detector.predict(headline)

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["prediction_code"], 1)  # Clickbait
        self.assertGreater(len(res["stylistic_cues"]), 0)

    def test_clickbait_detection_informative_pattern(self):
        headline = "Central Bank Holds Benchmark Interest Rate Steady at 3.5 Percent"
        res = self.cb_detector.predict(headline)

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["prediction_code"], 0)  # Non-clickbait

    def test_mismatch_detector(self):
        headline = "Apple Reports Record First Quarter Earnings on iPhone Demand"
        body = "Apple Inc. announced its financial results for its fiscal first quarter, with revenue surpassing market analyst expectations driven by strong global demand for its flagship smartphone devices."
        res = self.mm_detector.analyze(headline, body)

        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["similarity_score"], 0.10)
        self.assertIn("Apple", res["shared_keywords"])

    def test_mismatch_detector_swapped_content(self):
        headline = "Tragic Plane Crash Diverts International Flights in Zurich"
        body = "The local culinary festival celebrated organic vegetable farming with award-winning heirloom tomato tastings and pastry workshops hosted by neighborhood bakeries."
        res = self.mm_detector.analyze(headline, body)

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["status_code"], "mismatch")
        self.assertLess(res["similarity_score"], res["calibrated_threshold"])

    def test_explainability_synthesis(self):
        fn = self.fake_clf.predict("Test Headline", "Test Body")
        cb = self.cb_detector.predict("Test Headline")
        mm = self.mm_detector.analyze("Test Headline", "Test Body")
        ev = self.verifier.verify_claim("Test Headline")

        synth = ExplainabilityEngine.synthesize(fn, cb, mm, ev)
        self.assertIn("overall_badge", synth)
        self.assertIn("overall_summary", synth)
        self.assertIn("core_disclaimer", synth)


if __name__ == "__main__":
    unittest.main()
