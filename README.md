# NewsGuard: Explainable Fake News, Clickbait & Headline–Body Mismatch Detection System

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Framework-Flask](https://img.shields.io/badge/Framework-Flask_3.1-black.svg)](https://flask.palletsprojects.com/)
[![ML-Scikit--Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn_1.7-orange.svg)](https://scikit-learn.org/)
[![Evaluation-Held--Out%20Test](https://img.shields.io/badge/Evaluation-Held--Out%20Test%20(20%25)-green.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)]()

> **College Machine Learning Laboratory (MLM) Capstone Project**  
> An explainable, multi-dimensional news forensics engine combining linguistic veracity classification, clickbait stylometry, calibrated headline–body semantic consistency, and external encyclopedic verification.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Forensic Architecture & Methodology](#forensic-architecture--methodology)
3. [Verified Machine Learning Evaluation](#verified-machine-learning-evaluation)
4. [Dataset Provenance & Attribution](#dataset-provenance--attribution)
5. [System Directory Structure](#system-directory-structure)
6. [Quickstart & Installation](#quickstart--installation)
7. [Running the Application](#running-the-application)
8. [Automated Test Suite](#automated-test-suite)
9. [Faculty Demonstration Scenarios](#faculty-demonstration-scenarios)
10. [Scientific Boundaries & Limitations](#scientific-boundaries--limitations)
11. [Deployment Guide](#deployment-guide)
12. [Academic Literature & References](#academic-literature--references)

---

## 1. Project Overview

Digital media and social platforms are increasingly disrupted by three distinct but overlapping forms of misleading content:
1. **Fabricated Disinformation ("Fake News"):** Fabricated narratives mimicking authentic journalism to mislead readers or manipulate discourse.
2. **Clickbait Rhetoric:** Sensational, suspenseful, or exaggerated headlines engineered to monetize user attention.
3. **Headline–Body Mismatch (Bait-and-Switch):** Sensational headlines paired with routine, divergent, or completely unrelated body text.

Most existing automated checkers treat "fake news" as a single black-box classification problem. **NewsGuard** addresses this limitation by **disentangling the analysis into three independent machine learning modules** plus an **external knowledge grounding module**, providing transparent local feature attributions and mathematically grounded explanations.

---

## 2. Forensic Architecture & Methodology

NewsGuard's pipeline processes news articles through four stages:

```
                          ┌───────────────────────────┐
                          │   News Headline & Body    │
                          │   (User Input / Web URL)  │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │     NLP Preprocessing     │
                          │  (HTML strip, N-grams,    │
                          │   entity term extraction) │
                          └─────────────┬─────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                │                       │                       │
                ▼                       ▼                       ▼
    ┌──────────────────────┐┌──────────────────────┐┌──────────────────────┐
    │  Fake News Classifier││  Clickbait Detector  ││   Mismatch Detector  │
    │  TF-IDF + Linear SGD ││ TF-IDF + Multinomial ││ TF-IDF Cosine Sim.   │
    │  (SVM Loss Function) ││      Naive Bayes     ││ Calibrated τ = 0.100 │
    └───────────┬──────────┘└───────────┬──────────┘└───────────┬──────────┘
                │                       │                       │
                └───────────────────────┼───────────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │    External Verification  │
                          │  (Wikipedia OpenSearch    │
                          │   Entity Context Query)   │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   Explainability Engine   │
                          │ • Local Feature Weights   │
                          │ • Rhetorical Cues Highlight│
                          │ • Keyword Overlap Grounding│
                          │ • Ethical AI Disclaimers  │
                          └───────────────────────────┘
```

1. **Module 1: Fake News Classification (Linguistic Veracity)**  
   Transforms full article text using sublinear TF-IDF word and bigram vectorization (10,000 features). Classifies using calibrated Linear SGD (Support Vector Machine).
2. **Module 2: Clickbait Detection (Headline Stylometry)**  
   Analyzes headline syntax using TF-IDF bigrams (8,000 features) with Multinomial Naive Bayes, supplemented by rule-based detection for rhetorical listicles, suspense lures, and sensory intensifiers.
3. **Module 3: Headline–Body Mismatch Detection (Consistency Alignment)**  
   Measures cosine similarity between headline and body vector representations. Uses a validation-calibrated decision threshold ($\tau^* = 0.100$) to detect bait-and-switch divergence.
4. **Module 4: External Verification (Reference Grounding)**  
   Queries Wikipedia's public OpenSearch API for key named entities extracted from the headline, surfacing relevant encyclopedic references.
5. **Module 5: Multidimensional Explainability**  
   Computes local token attribution (positive and negative linear feature weights) for each specific article, highlights grounded shared entities, and enforces clear ethical boundaries.

---

## 3. Verified Machine Learning Evaluation

All models were evaluated on **strictly held-out test sets** (80% training, 20% test, stratified split, random seed `42`). **No metrics are fabricated or hardcoded.** Metrics are saved directly from the training script to `models/evaluation_results.json`.

### Task 1: Fake News Classification
- **Dataset**: George McIntire / Lutz Hamel Benchmark (6,295 cleaned articles: 3,171 REAL, 3,124 FAKE)
- **Held-Out Test Size**: 1,259 articles (634 REAL, 625 FAKE)

| Model Candidate | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | Selected Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SGD)** | **93.80%** | **93.83%** | **93.81%** | **93.80%** | **93.80%** | **Optimal** |
| Logistic Regression ($C=2.0$) | 93.73% | 93.75% | 93.73% | 93.72% | 93.72% | Evaluated |
| Multinomial Naive Bayes ($\alpha=0.1$) | 91.02% | 91.14% | 91.04% | 91.02% | 91.02% | Evaluated |

#### Confusion Matrix (SVM Held-Out Test, $N=1,259$):
- **True Real (Credible)**: 593 | **False Fake**: 41
- **False Real**: 37 | **True Fake (Unreliable)**: 588

---

### Task 2: Clickbait Detection
- **Dataset**: Chakraborty et al. (ASONAM 2016) Dataset (32,000 headlines: 16,001 Non-Clickbait, 15,999 Clickbait)
- **Held-Out Test Size**: 6,400 headlines (3,200 Non-Clickbait, 3,200 Clickbait)

| Model Candidate | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | Selected Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **96.05%** | **96.05%** | **96.05%** | **96.05%** | **96.05%** | **Optimal** |
| Logistic Regression ($C=2.0$) | 95.63% | 95.64% | 95.63% | 95.62% | 95.62% | Evaluated |
| Support Vector Machine (SGD) | 94.78% | 94.85% | 94.78% | 94.78% | 94.78% | Evaluated |

#### Confusion Matrix (Multinomial NB Held-Out Test, $N=6,400$):
- **True Non-Clickbait**: 3,083 | **False Clickbait**: 117
- **False Non-Clickbait**: 136 | **True Clickbait**: 3,064

---

### Task 3: Headline–Body Mismatch Detector
- **Methodology**: TF-IDF N-gram Vectorization with Cosine Similarity
- **Validation Tuning**: Decision threshold searched across $\tau \in [0.04, 0.35]$
- **Optimal Decision Threshold**: **$\tau^* = 0.100$**
- **Test Set Size**: 1,200 pairs (600 matched, 600 synthetically swapped)
- **Test Accuracy**: **90.58%**
- **Test Macro F1**: **90.55%**
- **Mean Matched Similarity**: $0.2185$ vs **Mean Mismatched Similarity**: $0.0381$

---

## 4. Dataset Provenance & Attribution

1. **Fake or Real News Dataset**  
   - *Curator*: George McIntire & Lutz Hamel (University of Rhode Island)  
   - *Source*: [https://github.com/lutzhamel/fake-news](https://github.com/lutzhamel/fake-news)  
   - *Description*: Real articles from mainstream wire services (Reuters, AP, NYT) and deceptive stories from debunked political domains.
2. **Stop Clickbait Dataset**  
   - *Authors*: Abhijnan Chakraborty, Bhargavi Paranjape, Sourya Kakarla, and Niloy Ganguly  
   - *Affiliation*: IIT Kharagpur & Max Planck Institute  
   - *Citation*: Proceedings of the 2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), pp. 9–16.  
   - *Source*: [https://github.com/bhargaviparanjape/clickbait](https://github.com/bhargaviparanjape/clickbait)

---

## 5. System Directory Structure

```
MLM-PROJECT/
├── app.py                      # Production Flask Web Server & REST API
├── requirements.txt            # Python dependencies
├── train_models.py             # Reproducible training & evaluation pipeline
├── README.md                   # System documentation & research report
├── data/
│   ├── README.md               # Dataset documentation, origins, and citations
│   └── raw/
│       ├── clickbait_data.gz   # Chakraborty et al. clickbait headlines
│       └── non_clickbait_data.gz
├── models/
│   ├── fake_news_model.joblib  # Trained SVM model pipeline
│   ├── clickbait_model.joblib  # Trained Naive Bayes model pipeline
│   ├── mismatch_detector.joblib# Calibrated mismatch threshold & vectorizer
│   └── evaluation_results.json # Verified held-out test evaluation export
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Text normalization & entity extraction
│   ├── fake_news_classifier.py # Veracity inference & local token attribution
│   ├── clickbait_detector.py   # Clickbait classification & stylometry cues
│   ├── mismatch_detector.py    # TF-IDF cosine similarity & threshold logic
│   ├── external_verifier.py    # OpenSearch encyclopedic query engine
│   └── explainability.py       # Multi-signal synthesis & disclaimers
├── static/
│   ├── css/
│   │   └── style.css           # Custom AI research dashboard stylesheet
│   ├── js/
│   │   ├── main.js             # Theme toggle & session state manager
│   │   ├── analyze.js          # Asynchronous analysis & forensics renderer
│   │   └── charts.js           # Chart.js outcomes & comparison charts
│   └── img/
│       └── logo.svg            # NewsGuard SVG identity icon
├── templates/
│   ├── base.html               # Shared layout, sidebar navigation, drawer
│   ├── index.html              # Page A: Overview Dashboard
│   ├── analyze.html            # Page B: News Forensics Analysis Page
│   ├── performance.html        # Page C: Model Performance & Benchmark Metrics
│   └── about.html              # Page D: Academic Architecture & References
└── tests/
    ├── test_preprocessing.py   # Normalization & validation tests
    ├── test_models.py          # ML inference & pipeline integrity tests
    ├── test_api.py             # Flask route & REST API integration tests
    └── test_scenarios.py       # Four required representative scenarios
```

---

## 6. Quickstart & Installation

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/sowmi-1409/MLM-PROJECT.git
cd MLM-PROJECT
```

### Step 2: Create and Activate Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. Running the Application

### Option A: Launch Pre-Trained Application Directly
Trained model weights and evaluation records are already included in `models/`. Simply run:
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### Option B: Retrain Models from Scratch (Deterministic Reproduction)
To re-download datasets, re-train candidate algorithms, and re-generate `models/evaluation_results.json`:
```bash
python train_models.py
```

---

## 8. Automated Test Suite

NewsGuard includes a full suite of automated unit, integration, and scenario tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

All 24 automated test cases evaluate:
- Text normalization, HTML cleaning, and entity extraction.
- Fake news inference, class probabilities, and feature weight extraction.
- Clickbait stylometric pattern matching and Naive Bayes probability ratios.
- Headline–body cosine similarity threshold discrimination on matched vs swapped articles.
- Flask endpoints, HTTP status codes, and input validation handling.
- The 4 representative demonstration scenarios.

---

## 9. Faculty Demonstration Scenarios

Four pre-curated test cases can be loaded with one click on the **News Analysis** page via the preset chips:

| Scenario | Input Headline | Input Body | Expected Forensic Output |
| :--- | :--- | :--- | :--- |
| **1. Credible News** | *NASA James Webb Space Telescope Reveals Details of Exoplanet Atmosphere* | Detailed spectroscopic observations from NASA symposium. | **Credible Journalism (93.8% confidence)**, Non-Clickbait, Consistent ($\text{Sim} > 0.100$), Public Corroboration. |
| **2. Sensational Clickbait** | *You Won't Believe What These 7 Shocking Tech Companies Are Hiding!!* | Standard SEC quarterly compliance filings. | **Clickbait Warning (96.1% confidence)**, Listicle & hyperbole flags, informative body, Ethical AI Notice displayed. |
| **3. Fabricated Disinformation** | *Secret Elite Underground Gathering Caught Enacting Massive Coverup* | Conspiratorial claims regarding clandestine rituals and hidden cures. | **Unreliable / Fake Flag**, speculative vocabulary attribution, absence of external encyclopedic corroboration. |
| **4. Headline–Body Mismatch** | *Hollywood Megastar Arrested in Shocking Midnight Narcotics Raid* | European Central Bank interest rate adjustments in Frankfurt. | **Headline–Body Mismatch Warning**, $\text{Cosine Sim} < 0.100$, Unmatched entity list, Bait-and-switch alert. |

---

## 10. Scientific Boundaries & Limitations

1. **Statistical Correlation vs Truth**: Machine learning models assess linguistic structure, vocabulary distribution, and stylistic cues. They cannot replace physical real-world fact verification.
2. **Sensationalism $\neq$ Falsity**: Legitimate news organizations may occasionally use curiosity-inducing headlines; conversely, disinformation may be phrased in a calm, academic tone.
3. **Domain & Temporal Drift**: Models trained on 2016 political benchmarks exhibit higher confidence on political topics than on sports, localized crime, or entertainment journalism.
4. **Lexical Mismatch**: Cosine similarity measures word and phrase overlap. Breaking stories that use novel metaphors may show low lexical overlap while still being topically aligned.

---

## 11. Deployment Guide

The application is lightweight and ready for deployment to free or low-cost cloud platforms:

### Deploy to Render
1. Create a free account at [render.com](https://render.com).
2. Connect your GitHub repository `https://github.com/sowmi-1409/MLM-PROJECT`.
3. Select **Web Service**, set **Environment** to `Python 3`.
4. Set **Build Command**: `pip install -r requirements.txt`
5. Set **Start Command**: `gunicorn app:app` (or `python app.py`)

### Deploy to Hugging Face Spaces
1. Create a Space with the **Docker** or **Gradio/Flask** SDK.
2. Push repository files.
3. Map internal port `5000` to the container port.

---

## 12. Academic Literature & References

1. **Chakraborty, A., Paranjape, B., Kakarla, S., & Ganguly, N. (2016).**  
   *Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media.*  
   Proceedings of the 2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), pp. 9–16.
2. **Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017).**  
   *Fake News Detection on Social Media: A Data Mining Perspective.*  
   ACM SIGKDD Explorations Newsletter, 19(1), 22–36.
3. **Wang, W. Y. (2017).**  
   *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.*  
   Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 422–426.
4. **Ferreira, W., & Vlachos, A. (2016).**  
   *Emergent: A Novel Dataset for Stance Classification.*  
   Proceedings of NAACL-HLT 2016, pp. 1163–1168.
5. **Hamel, L., & McIntire, G. (2017).**  
   *Fake News Dataset & Analysis.*  
   University of Rhode Island Department of Computer Science and Statistics.
