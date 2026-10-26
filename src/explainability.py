"""
Explainability & Synthesis Engine for NewsGuard
Aggregates multidimensional signals (veracity, clickbait, headline-body alignment, external context)
and generates transparent, mathematically grounded explanations.
"""

from typing import Dict, Any, List


class ExplainabilityEngine:
    @staticmethod
    def synthesize(
        fake_result: Dict[str, Any],
        clickbait_result: Dict[str, Any],
        mismatch_result: Dict[str, Any],
        verifier_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize explainable findings into an overarching report.
        """
        risk_flags = []
        positive_signals = []

        # 1. Fake News Signal
        if fake_result.get("prediction_code") == 1:
            risk_flags.append({
                "type": "veracity_risk",
                "severity": "high",
                "title": "Unreliable Linguistic Markers Detected",
                "description": f"Statistical model detected stylistic markers associated with disinformation ({fake_result.get('confidence_percent', 0)}% confidence)."
            })
        else:
            positive_signals.append({
                "type": "veracity_credible",
                "title": "Credible Journalism Indicators",
                "description": f"Vocabulary and sentence patterns align with credible news reporting ({fake_result.get('confidence_percent', 0)}% confidence)."
            })

        # 2. Clickbait Signal
        if clickbait_result.get("prediction_code") == 1:
            cues = clickbait_result.get("stylistic_cues", [])
            cue_text = f" Patterns identified: {', '.join([c['pattern'] for c in cues])}" if cues else ""
            risk_flags.append({
                "type": "clickbait_warning",
                "severity": "medium",
                "title": "Sensational or Clickbait Headline",
                "description": f"Headline employs emotional or curiosity-gap phrasing.{cue_text}"
            })
        else:
            positive_signals.append({
                "type": "headline_informative",
                "title": "Informative Headline",
                "description": "Headline directly describes the core event without sensational embellishment."
            })

        # 3. Mismatch Signal
        status_code = mismatch_result.get("status_code", "unknown")
        if status_code == "mismatch":
            risk_flags.append({
                "type": "mismatch_alert",
                "severity": "medium",
                "title": "Headline–Body Mismatch Warning",
                "description": (
                    f"Low semantic similarity ({mismatch_result.get('similarity_score', 0):.3f} vs threshold "
                    f"{mismatch_result.get('calibrated_threshold', 0.10):.3f}). Headline may exaggerate or diverge from body claims."
                )
            })
        elif status_code == "consistent":
            positive_signals.append({
                "type": "mismatch_aligned",
                "title": "Headline–Body Alignment Confirmed",
                "description": f"High lexical overlap and topic consistency (Similarity: {mismatch_result.get('similarity_score', 0):.3f})."
            })

        # 4. External Verification
        verdict = verifier_result.get("verdict", "")
        if "Found" in verdict:
            positive_signals.append({
                "type": "external_corroborated",
                "title": "External Public Corroboration",
                "description": "Public records and encyclopedia entries corroborate the main entities or topics."
            })
        elif "Insufficient" in verdict:
            risk_flags.append({
                "type": "external_unverified",
                "severity": "low",
                "title": "Unverified Public Entities",
                "description": "No corresponding Wikipedia encyclopedic entries were retrieved for the primary search query."
            })

        # Overall Assessment Matrix
        if len(risk_flags) == 0:
            overall_badge = "success"
            overall_summary = "High Credibility Profile: Content aligns with standard journalism across all evaluated dimensions."
        elif len(risk_flags) >= 2 and fake_result.get("prediction_code") == 1:
            overall_badge = "danger"
            overall_summary = "High Caution Advised: Multiple risk factors detected including unreliable linguistic markers."
        else:
            overall_badge = "warning"
            overall_summary = "Moderate Caution Advised: Selected warnings detected (e.g. sensational headline or stylistic discrepancies)."

        # Crucial academic disclaimer
        core_disclaimer = (
            "Ethical AI Notice: NewsGuard produces machine-learning probability assessments based on lexical patterns, "
            "cosine similarity, and open retrieval. A low similarity score or sensational headline does NOT automatically "
            "prove that an article is factually fraudulent. Always verify claims against reputable primary sources."
        )

        return {
            "overall_badge": overall_badge,
            "overall_summary": overall_summary,
            "risk_flags": risk_flags,
            "positive_signals": positive_signals,
            "core_disclaimer": core_disclaimer
        }
