"""
Test Suite for Required Representative Scenarios:
1. Consistent headline and body
2. Sensational but potentially accurate headline
3. Unrelated headline and article body (mismatch)
4. Claim requiring external verification
"""

import unittest
from app import app


class TestRepresentativeScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_scenario_1_consistent_headline_and_body(self):
        """Case 1: Consistent headline and body with high alignment and credible markers."""
        payload = {
            "headline": "NASA James Webb Space Telescope Reveals Unprecedented Details of Distant Exoplanet Atmosphere",
            "body": "Astronomers using NASA's James Webb Space Telescope have captured clear spectroscopic signatures of carbon dioxide and water vapor in the atmosphere of an exoplanet orbiting a sun-like star hundreds of light-years away. The discovery was announced today during an international scientific symposium in Washington, D.C."
        }
        res = self.client.post("/api/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["results"]

        # Headline-body should be consistent
        self.assertEqual(data["mismatch"]["status_code"], "consistent")
        self.assertGreaterEqual(data["mismatch"]["similarity_score"], data["mismatch"]["calibrated_threshold"])
        self.assertEqual(data["clickbait"]["prediction_code"], 0)  # Non-clickbait

    def test_scenario_2_sensational_headline_with_real_body(self):
        """Case 2: Sensational headline with standard body (should flag clickbait but not automatically fake)."""
        payload = {
            "headline": "You Won't Believe What These 7 Shocking Tech Companies Are Hiding From Everyone!!",
            "body": "Major technology corporations submitted their required quarterly compliance filings to federal regulators this past Tuesday. The quarterly reports, which are routine public disclosure documents under SEC guidelines, document standard operational capital expenditures, software licensing revisions, and server infrastructure costs across multiple cloud server facilities."
        }
        res = self.client.post("/api/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["results"]

        # Clickbait should flag warning
        self.assertEqual(data["clickbait"]["prediction_code"], 1)
        self.assertGreater(len(data["clickbait"]["stylistic_cues"]), 0)
        # Should NOT automatically assert that real body is fabricated
        self.assertIn("Ethical AI Notice", data["synthesis"]["core_disclaimer"])

    def test_scenario_3_unrelated_headline_and_body(self):
        """Case 3: Unrelated headline and body (bait-and-switch discrepancy)."""
        payload = {
            "headline": "Hollywood Megastar Arrested in Shocking Midnight Narcotics Raid at Private Villa",
            "body": "The European Central Bank announced an increase of 25 basis points in benchmark lending rates following its monthly governing council meeting in Frankfurt. Commercial banking institutions across the Eurozone noted that the policy adjustment aims to anchor medium-term inflation expectations amid elevated manufacturing energy expenses and persistent agricultural supply chain bottlenecks."
        }
        res = self.client.post("/api/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["results"]

        # Mismatch detector should flag discrepancy
        self.assertEqual(data["mismatch"]["status_code"], "mismatch")
        self.assertLess(data["mismatch"]["similarity_score"], data["mismatch"]["calibrated_threshold"])

    def test_scenario_4_claim_requiring_external_verification(self):
        """Case 4: Specific named entities / claim queried against external knowledge."""
        payload = {
            "headline": "James Webb Space Telescope Exoplanet Atmosphere Observations",
            "body": "Scientific teams analyzed observational spectra obtained from the NIRSpec instrument on the James Webb Space Telescope."
        }
        res = self.client.post("/api/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["results"]

        self.assertIn("external_verification", data)
        self.assertIn("verdict", data["external_verification"])
        self.assertIn("query_used", data["external_verification"])


if __name__ == "__main__":
    unittest.main()
