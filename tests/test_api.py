"""
Integration Tests for Flask Web Application & REST API
"""

import unittest
import json
from app import app


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_pages_render_status_200(self):
        for route in ["/", "/analyze", "/performance", "/about"]:
            res = self.client.get(route)
            self.assertEqual(res.status_code, 200, f"Route {route} failed to render.")

    def test_api_metrics_endpoint(self):
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("fake_news_task", data["evaluation"])
        self.assertIn("clickbait_task", data["evaluation"])
        self.assertIn("headline_body_mismatch_task", data["evaluation"])

    def test_api_samples_endpoint(self):
        res = self.client.get("/api/samples")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["samples"]), 4)

    def test_api_analyze_valid_input(self):
        payload = {
            "headline": "WHO Releases Updated Global Public Health Guidelines on Clean Water",
            "body": "The World Health Organization published revised standards for municipal drinking water filtration systems, emphasizing microbiological safety and heavy metal removal across developing regions.",
            "url": ""
        }
        res = self.client.post("/api/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("fake_news", data["results"])
        self.assertIn("clickbait", data["results"])
        self.assertIn("mismatch", data["results"])
        self.assertIn("external_verification", data["results"])
        self.assertIn("synthesis", data["results"])

    def test_api_analyze_empty_headline_error(self):
        payload = {
            "headline": "",
            "body": "Some body text here that is sufficiently long for analysis."
        }
        res = self.client.post("/api/analyze", json=payload)
        self.assertEqual(res.status_code, 422)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_api_session_stats_and_clear(self):
        res = self.client.get("/api/session-stats")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])

        clear_res = self.client.post("/api/clear-session")
        self.assertEqual(clear_res.status_code, 200)
        clear_data = clear_res.get_json()
        self.assertTrue(clear_data["success"])


if __name__ == "__main__":
    unittest.main()
