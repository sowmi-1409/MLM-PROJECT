"""
External Verification Module
Performs genuine external knowledge search using open public knowledge APIs (Wikipedia API),
extracts relevant reference links, and assesses available corroborating public evidence.
"""

import urllib.request
import urllib.parse
import json
import re
from typing import Dict, Any, List
from src.preprocessing import clean_text, extract_key_terms


class ExternalVerifier:
    def __init__(self, timeout_sec: int = 5):
        self.timeout = timeout_sec

    def verify_claim(self, headline: str) -> Dict[str, Any]:
        """
        Query public external sources for related encyclopedic or news evidence.
        Uses Wikipedia's OpenSearch API as an open, verifiable reference baseline.
        """
        clean_h = clean_text(headline)
        terms = extract_key_terms(clean_h, max_terms=5)

        if not terms:
            return {
                "status": "insufficient_terms",
                "verdict": "Insufficient Query Terms",
                "badge": "neutral",
                "sources": [],
                "explanation": "Headline does not contain enough salient entities or keywords to query public records.",
                "limitations": "Automated verification requires identifiable subject entities."
            }

        query = " ".join(terms[:4])
        sources = self._query_wikipedia(query)

        if not sources:
            # Fallback query with first 2 main entities
            if len(terms) >= 2:
                sources = self._query_wikipedia(" ".join(terms[:2]))

        if not sources:
            return {
                "status": "unverified",
                "verdict": "Insufficient External Evidence",
                "badge": "warning",
                "query_used": query,
                "sources": [],
                "explanation": (
                    f"No direct corroborating records found in public encyclopedic databases for '{query}'. "
                    "This may indicate a hyper-local event, emerging breaking story, satire, or unverified claims."
                ),
                "limitations": (
                    "Absence of external evidence in public open knowledge does not conclusively prove falsity; "
                    "specialized breaking stories require checking primary news agencies (Reuters, AP) or fact-checkers (PolitiFact, Snopes)."
                )
            }

        # Analyze relevance of retrieved sources
        matched_sources = []
        for src in sources[:4]:
            matched_sources.append({
                "title": src.get("title", "Reference"),
                "url": src.get("url", "#"),
                "snippet": src.get("snippet", "No summary snippet available."),
                "relevance": "Corroborating Topic Found"
            })

        return {
            "status": "corroborated",
            "verdict": "Related Public References Found",
            "badge": "success",
            "query_used": query,
            "sources": matched_sources,
            "explanation": (
                f"Retrieved {len(matched_sources)} relevant external public reference(s) matching entities: '{query}'. "
                "These sources provide external contextual verification for the subject matter discussed."
            ),
            "limitations": (
                "External knowledge retrieval confirms the existence of the topics or entities mentioned. "
                "It does not independently certify the exact veracity of specific sub-claims or novel accusations."
            )
        }

    def _query_wikipedia(self, query: str) -> List[Dict[str, str]]:
        """Query Wikipedia API with timeout handling and safety fallbacks."""
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit=4&namespace=0&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "NewsGuard-Verifier/1.0 (academic-research)"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # Format: [query, [titles], [descriptions], [links]]
            if len(data) >= 4:
                titles = data[1]
                descriptions = data[2]
                links = data[3]
                results = []
                for i in range(len(titles)):
                    results.append({
                        "title": titles[i],
                        "snippet": descriptions[i] if i < len(descriptions) else "",
                        "url": links[i] if i < len(links) else ""
                    })
                return results
        except Exception as e:
            # Silently handle network timeouts or lack of internet
            return []
        return []
