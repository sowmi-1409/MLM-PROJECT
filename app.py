"""
NewsGuard: Explainable Fake News, Clickbait & Headline-Body Mismatch Detection System
=====================================================================================
Production Flask Application & REST API
"""

import os
import json
import re
import urllib.request
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup

from src.preprocessing import clean_text, validate_input
from src.fake_news_classifier import FakeNewsClassifier
from src.clickbait_detector import ClickbaitDetector
from src.mismatch_detector import MismatchDetector
from src.external_verifier import ExternalVerifier
from src.explainability import ExplainabilityEngine

# Initialize Flask
app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = os.urandom(24)

# Initialize ML Classifiers
print("[INIT] Loading NewsGuard Machine Learning Engine...")
fake_classifier = FakeNewsClassifier()
clickbait_detector = ClickbaitDetector()
mismatch_detector = MismatchDetector()
external_verifier = ExternalVerifier(timeout_sec=4)

# Load real evaluation results
EVAL_PATH = os.path.join(os.path.dirname(__file__), "models", "evaluation_results.json")
evaluation_data = {}
if os.path.exists(EVAL_PATH):
    try:
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            evaluation_data = json.load(f)
        print("[INIT] Model evaluation metrics successfully loaded.")
    except Exception as e:
        print(f"[INIT] Warning: Could not parse evaluation_results.json: {e}")
else:
    print("[INIT] Warning: evaluation_results.json not found. Run train_models.py first.")

# In-memory session tracking for faculty demonstration
session_stats = {
    "total_analyzed": 0,
    "fake_count": 0,
    "clickbait_count": 0,
    "mismatch_count": 0,
    "credible_count": 0,
    "history": []
}

# Pre-curated representative demonstration samples
SAMPLE_ARTICLES = [
    {
        "id": "credible_science",
        "title": "Credible Journalism: NASA James Webb Telescope Discovery",
        "headline": "NASA James Webb Space Telescope Reveals Unprecedented Details of Distant Exoplanet Atmosphere",
        "body": "Astronomers using NASA's James Webb Space Telescope have captured the first clear chemical evidence of carbon dioxide and water vapor in the atmosphere of an exoplanet orbiting a sun-like star hundreds of light-years away. The discovery was announced today during an international scientific symposium in Washington, D.C. Researchers highlighted that these observations provide critical spectroscopic data that will help planetary scientists characterize atmospheric compositions across different solar systems.",
        "category": "Science & Technology",
        "expected": "Real news, informative headline, high consistency."
    },
    {
        "id": "sensational_clickbait",
        "title": "Sensational Clickbait Headline with Real Body",
        "headline": "You Won't Believe What These 7 Shocking Tech Companies Are Hiding From Everyone!!",
        "body": "Major technology corporations submitted their required quarterly compliance filings to federal regulators this past Tuesday. The quarterly reports, which are routine public disclosure documents under SEC guidelines, document standard operational capital expenditures, software licensing revisions, and server infrastructure costs across multiple cloud server facilities.",
        "category": "Technology & Business",
        "expected": "Clickbait warning, informative body, slight sensationalism disparity."
    },
    {
        "id": "fake_news_hoax",
        "title": "Fabricated Disinformation Hoax",
        "headline": "Secret Elite Underground Gathering Caught Enacting Massive Coverup Scheme Across Nation",
        "body": "Shocking whistleblower leaks obtained by underground investigators reveal an elaborate conspiratorial plot orchestrated by shadowy international globalist cabals operating completely in secret. Top officials were purportedly photographed holding private clandestine midnight rituals to manipulate elections and suppress undeniable biological cures that mainstream media refuses to disclose.",
        "category": "Disinformation / Conspiracy",
        "expected": "Fake news flag, sensational cues, speculative vocabulary."
    },
    {
        "id": "headline_body_mismatch",
        "title": "Headline–Body Mismatch (Bait-and-Switch)",
        "headline": "Hollywood Megastar Arrested in Shocking Midnight Narcotics Raid at Private Villa",
        "body": "The European Central Bank announced an increase of 25 basis points in benchmark lending rates following its monthly governing council meeting in Frankfurt. Commercial banking institutions across the Eurozone noted that the policy adjustment aims to anchor medium-term inflation expectations amid elevated manufacturing energy expenses and persistent agricultural supply chain bottlenecks.",
        "category": "Financial Report vs Scandal Headline",
        "expected": "Severe headline-body discrepancy, low cosine similarity."
    }
]


# =====================================================================
# HTML Page Routes
# =====================================================================

@app.route("/")
def index():
    """Page A: Overview Dashboard"""
    return render_template(
        "index.html",
        active_page="index",
        stats=session_stats,
        evaluation=evaluation_data
    )


@app.route("/analyze")
def analyze():
    """Page B: News Analysis Page"""
    return render_template(
        "analyze.html",
        active_page="analyze",
        samples=SAMPLE_ARTICLES
    )


@app.route("/performance")
def performance():
    """Page C: Model Performance Page"""
    return render_template(
        "performance.html",
        active_page="performance",
        eval_data=evaluation_data
    )


@app.route("/about")
def about():
    """Page D: About Project Page"""
    return render_template(
        "about.html",
        active_page="about"
    )


# =====================================================================
# REST API Endpoints
# =====================================================================

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Comprehensive news analysis endpoint.
    Processes headline, body, and optional URL across all 4 machine learning pipelines.
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON or form payload."}), 400

    headline = data.get("headline", "").strip()
    body = data.get("body", "").strip()
    article_url = data.get("url", "").strip()

    # Input validation
    is_valid, err_msg = validate_input(headline, body)
    if not is_valid:
        return jsonify({"success": False, "error": err_msg}), 422

    # 1. Run Machine Learning Classifiers
    fake_res = fake_classifier.predict(headline, body)
    clickbait_res = clickbait_detector.predict(headline)
    mismatch_res = mismatch_detector.analyze(headline, body)
    verifier_res = external_verifier.verify_claim(headline)

    # 2. Explainability & Synthesis
    synthesis = ExplainabilityEngine.synthesize(
        fake_res,
        clickbait_res,
        mismatch_res,
        verifier_res
    )

    # 3. Update Session Statistics
    session_stats["total_analyzed"] += 1
    if fake_res.get("prediction_code") == 1:
        session_stats["fake_count"] += 1
    else:
        session_stats["credible_count"] += 1

    if clickbait_res.get("prediction_code") == 1:
        session_stats["clickbait_count"] += 1

    if mismatch_res.get("status_code") == "mismatch":
        session_stats["mismatch_count"] += 1

    # Record in recent history (store last 10 analyses)
    history_entry = {
        "id": session_stats["total_analyzed"],
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "headline": headline[:80] + ("..." if len(headline) > 80 else ""),
        "veracity_label": fake_res.get("label", "Unknown"),
        "veracity_code": fake_res.get("prediction_code", 0),
        "clickbait_code": clickbait_res.get("prediction_code", 0),
        "mismatch_code": 1 if mismatch_res.get("status_code") == "mismatch" else 0,
        "overall_badge": synthesis.get("overall_badge", "info")
    }
    session_stats["history"].insert(0, history_entry)
    if len(session_stats["history"]) > 10:
        session_stats["history"].pop()

    return jsonify({
        "success": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "headline": headline,
            "body_length": len(body.split()),
            "url": article_url
        },
        "results": {
            "fake_news": fake_res,
            "clickbait": clickbait_res,
            "mismatch": mismatch_res,
            "external_verification": verifier_res,
            "synthesis": synthesis
        },
        "session_stats": {
            "total_analyzed": session_stats["total_analyzed"],
            "fake_count": session_stats["fake_count"],
            "clickbait_count": session_stats["clickbait_count"],
            "mismatch_count": session_stats["mismatch_count"],
            "credible_count": session_stats["credible_count"]
        }
    })


@app.route("/api/fetch-url", methods=["POST"])
def api_fetch_url():
    """
    Fetch and scrape news article content from a user-supplied URL.
    Extracts page headline (H1/title) and body paragraphs.
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form
        url = data.get("url", "").strip()
    except Exception:
        return jsonify({"success": False, "error": "Invalid request payload."}), 400

    if not url:
        return jsonify({"success": False, "error": "Please provide a valid article URL."}), 400

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            html_content = response.read().decode("utf-8", errors="ignore")

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove irrelevant noise tags
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
            element.extract()

        # Extract title / headline
        title = ""
        h1 = soup.find("h1")
        if h1 and len(h1.get_text(strip=True)) > 5:
            title = h1.get_text(strip=True)
        elif soup.title and len(soup.title.get_text(strip=True)) > 5:
            title = soup.title.get_text(strip=True)
            # Remove site branding e.g. "Headline - Reuters"
            title = re.sub(r"\s*[-|–]\s*.*$", "", title).strip()

        # Extract body paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
        body = "\n\n".join(paragraphs[:12])  # Take up to 12 main paragraphs

        if not title and not body:
            return jsonify({
                "success": False,
                "error": "Could not automatically parse article content from this URL. Please copy and paste the text manually."
            }), 422

        return jsonify({
            "success": True,
            "url": url,
            "headline": title,
            "body": body
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to fetch content from URL: {str(e)}. The site may block automated access. Please copy and paste text manually."
        }), 400


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Retrieve sample pre-loaded news articles for quick faculty demonstration."""
    return jsonify({"success": True, "samples": SAMPLE_ARTICLES})


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return genuine model evaluation metrics exported during training."""
    if not evaluation_data:
        return jsonify({"success": False, "error": "Evaluation metrics not available on disk."}), 404
    return jsonify({"success": True, "evaluation": evaluation_data})


@app.route("/api/session-stats", methods=["GET"])
def api_session_stats():
    """Return current session statistics and recent history."""
    return jsonify({"success": True, "stats": session_stats})


@app.route("/api/clear-session", methods=["POST"])
def api_clear_session():
    """Reset session statistics for demonstration purposes."""
    global session_stats
    session_stats = {
        "total_analyzed": 0,
        "fake_count": 0,
        "clickbait_count": 0,
        "mismatch_count": 0,
        "credible_count": 0,
        "history": []
    }
    return jsonify({"success": True, "message": "Session statistics cleared."})


# =====================================================================
# Main Application Launcher
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NewsGuard System Online")
    print("Access the web interface at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
