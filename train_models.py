"""
NewsGuard Model Training & Evaluation Pipeline
==============================================
Reproducible training and evaluation script for:
1. Fake News Classification (George McIntire / Hamel Benchmark Dataset)
2. Clickbait Detection (Chakraborty et al. ASONAM 2016 Dataset)
3. Headline-Body Mismatch Detection (TF-IDF Cosine Similarity with Calibrated Threshold)

Outputs:
- Trained serialized models in `models/`
- Full genuine evaluation metrics in `models/evaluation_results.json`
"""

import os
import json
import gzip
import urllib.request
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.metrics.pairwise import cosine_similarity

# Constants & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RANDOM_STATE = 42

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def download_file(url: str, dest_path: str):
    """Download a file with requests chunking if not already present."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[CACHE] Found existing: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
        return
    print(f"[DOWNLOAD] Fetching {url} -> {dest_path}")
    import requests
    headers = {"User-Agent": "Mozilla/5.0 NewsGuard-ML-Pipeline"}
    r = requests.get(url, stream=True, timeout=60, headers=headers)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print(f"[DOWNLOAD] Completed: {dest_path} ({os.path.getsize(dest_path):,} bytes)")


def prepare_fake_news_data():
    """Download and prepare the Fake or Real News dataset."""
    url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
    dest = os.path.join(DATA_RAW_DIR, "fake_or_real_news.csv")
    download_file(url, dest)

    df = pd.read_csv(dest)
    print(f"[FAKE NEWS] Raw dataset loaded: {df.shape[0]} rows, columns={list(df.columns)}")

    # Clean text and handle missing values
    df = df.dropna(subset=["title", "text", "label"])
    df["title"] = df["title"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[(df["text"].str.len() > 30) & (df["title"].str.len() > 3)]

    # Normalize labels: FAKE -> 1, REAL -> 0
    df["label_num"] = df["label"].map({"FAKE": 1, "REAL": 0})
    df = df.dropna(subset=["label_num"])
    df["label_num"] = df["label_num"].astype(int)

    # Combine title and text for rich content classification
    df["full_text"] = df["title"] + " — " + df["text"]

    class_counts = df["label"].value_counts().to_dict()
    print(f"[FAKE NEWS] Cleaned samples: {len(df)} | Class distribution: {class_counts}")
    return df


def prepare_clickbait_data():
    """Download and prepare the Chakraborty et al. Clickbait dataset."""
    cb_url = "https://raw.githubusercontent.com/bhargaviparanjape/clickbait/master/dataset/clickbait_data.gz"
    non_cb_url = "https://raw.githubusercontent.com/bhargaviparanjape/clickbait/master/dataset/non_clickbait_data.gz"

    cb_dest = os.path.join(DATA_RAW_DIR, "clickbait_data.gz")
    non_cb_dest = os.path.join(DATA_RAW_DIR, "non_clickbait_data.gz")

    download_file(cb_url, cb_dest)
    download_file(non_cb_url, non_cb_dest)

    with gzip.open(cb_dest, "rt", encoding="utf-8", errors="ignore") as f:
        cb_lines = [line.strip() for line in f if len(line.strip()) > 5]

    with gzip.open(non_cb_dest, "rt", encoding="utf-8", errors="ignore") as f:
        non_cb_lines = [line.strip() for line in f if len(line.strip()) > 5]

    df_cb = pd.DataFrame({"headline": cb_lines, "label": 1, "label_name": "Clickbait"})
    df_non = pd.DataFrame({"headline": non_cb_lines, "label": 0, "label_name": "Non-Clickbait"})

    df = pd.concat([df_cb, df_non], ignore_index=True).drop_duplicates(subset=["headline"])
    class_counts = df["label_name"].value_counts().to_dict()
    print(f"[CLICKBAIT] Cleaned samples: {len(df)} | Class distribution: {class_counts}")
    return df


def evaluate_models(X_train, X_test, y_train, y_test, class_names, task_name="Classification"):
    """
    Train and rigorously evaluate candidate models:
    1. Logistic Regression
    2. Multinomial Naive Bayes
    3. Linear SGD / SVM Classifier
    """
    print(f"\n--- Training Candidate Models for {task_name} ---")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidate_models = {
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, random_state=RANDOM_STATE),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Support Vector Machine (SGD)": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }

    results = {}
    best_f1 = -1
    best_name = None
    best_model = None

    for name, clf in candidate_models.items():
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision_weighted": round(float(prec_w), 4),
            "recall_weighted": round(float(rec_w), 4),
            "f1_weighted": round(float(f1_w), 4),
            "confusion_matrix": cm,
            "classification_report": report
        }

        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec_macro:.4f} | Rec: {rec_macro:.4f} | F1: {f1_macro:.4f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_name = name
            best_model = clf

    # Extract top influential features for explainability
    top_features = {}
    if hasattr(best_model, "coef_"):
        feature_names = np.array(vectorizer.get_feature_names_out())
        coef = best_model.coef_[0]
        # Top positive (class 1: Fake / Clickbait)
        top_pos_idx = np.argsort(coef)[-25:][::-1]
        # Top negative (class 0: Real / Non-Clickbait)
        top_neg_idx = np.argsort(coef)[:25]

        top_features = {
            "class_1_indicators": [{"feature": feature_names[i], "weight": round(float(coef[i]), 3)} for i in top_pos_idx],
            "class_0_indicators": [{"feature": feature_names[i], "weight": round(float(coef[i]), 3)} for i in top_neg_idx]
        }

    return {
        "models": results,
        "best_model_name": best_name,
        "best_pipeline": {"vectorizer": vectorizer, "classifier": best_model},
        "top_features": top_features
    }


def train_and_evaluate_mismatch(fake_news_df):
    """
    Train and evaluate the Headline-Body Mismatch detector on paired vs mismatched articles.
    Calculates TF-IDF Cosine Similarity and tunes threshold to discriminate genuine matches from swapped pairs.
    """
    print("\n--- Calibrating Headline-Body Mismatch Detector ---")
    # Take real news articles with both title and substantial body
    sample_df = fake_news_df[fake_news_df["label"] == "REAL"].copy().reset_index(drop=True)
    if len(sample_df) > 2000:
        sample_df = sample_df.sample(n=2000, random_state=RANDOM_STATE).reset_index(drop=True)

    titles = sample_df["title"].tolist()
    bodies = sample_df["text"].tolist()

    # Split into train/validation and test pairs
    n_total = len(titles)
    split_idx = int(0.7 * n_total)

    train_titles, train_bodies = titles[:split_idx], bodies[:split_idx]
    test_titles, test_bodies = titles[split_idx:], bodies[split_idx:]

    # Fit TF-IDF on all text
    mismatch_vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2))
    mismatch_vectorizer.fit(train_titles + train_bodies)

    # Compute matched similarities on validation
    train_t_vec = mismatch_vectorizer.transform(train_titles)
    train_b_vec = mismatch_vectorizer.transform(train_bodies)
    matched_sims_val = np.array([cosine_similarity(train_t_vec[i], train_b_vec[i])[0][0] for i in range(len(train_titles))])

    # Generate shuffled/mismatched pairs (shift body by 1 to create mismatched negative pairs)
    shuffled_b_vec = mismatch_vectorizer.transform(train_bodies[1:] + [train_bodies[0]])
    mismatched_sims_val = np.array([cosine_similarity(train_t_vec[i], shuffled_b_vec[i])[0][0] for i in range(len(train_titles))])

    # Search optimal threshold on validation set (maximizing F1 score for match detection)
    threshold_candidates = np.linspace(0.04, 0.35, 32)
    best_thresh = 0.12
    best_f1 = -1

    val_y_true = np.concatenate([np.ones_like(matched_sims_val), np.zeros_like(mismatched_sims_val)])
    val_scores = np.concatenate([matched_sims_val, mismatched_sims_val])

    for th in threshold_candidates:
        val_y_pred = (val_scores >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(val_y_true, val_y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

    # Evaluate on held-out test pairs
    test_t_vec = mismatch_vectorizer.transform(test_titles)
    test_b_vec = mismatch_vectorizer.transform(test_bodies)
    matched_sims_test = np.array([cosine_similarity(test_t_vec[i], test_b_vec[i])[0][0] for i in range(len(test_titles))])

    shuffled_test_b_vec = mismatch_vectorizer.transform(test_bodies[1:] + [test_bodies[0]])
    mismatched_sims_test = np.array([cosine_similarity(test_t_vec[i], shuffled_test_b_vec[i])[0][0] for i in range(len(test_titles))])

    test_y_true = np.concatenate([np.ones_like(matched_sims_test), np.zeros_like(mismatched_sims_test)])
    test_scores = np.concatenate([matched_sims_test, mismatched_sims_test])
    test_y_pred = (test_scores >= best_thresh).astype(int)

    acc = accuracy_score(test_y_true, test_y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(test_y_true, test_y_pred).tolist()
    report = classification_report(test_y_true, test_y_pred, target_names=["Mismatched", "Consistent"], output_dict=True, zero_division=0)

    print(f"[MISMATCH DETECTOR] Calibrated Threshold: {best_thresh:.3f} | Test Acc: {acc:.4f} | F1: {f1:.4f}")

    mismatch_data = {
        "calibrated_threshold": round(float(best_thresh), 3),
        "mean_matched_similarity": round(float(np.mean(matched_sims_test)), 4),
        "mean_mismatched_similarity": round(float(np.mean(mismatched_sims_test)), 4),
        "test_accuracy": round(float(acc), 4),
        "test_precision": round(float(prec), 4),
        "test_recall": round(float(rec), 4),
        "test_f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "test_samples_total": len(test_y_true)
    }

    mismatch_pipeline = {
        "vectorizer": mismatch_vectorizer,
        "threshold": best_thresh,
        "metadata": mismatch_data
    }

    return mismatch_pipeline, mismatch_data


def main():
    print("=" * 65)
    print("NewsGuard Machine Learning Pipeline — Full Training & Evaluation")
    print("=" * 65)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    # 1. Fake News Dataset & Training
    df_fake = prepare_fake_news_data()
    X_train_fn, X_test_fn, y_train_fn, y_test_fn = train_test_split(
        df_fake["full_text"],
        df_fake["label_num"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_fake["label_num"]
    )

    fake_eval = evaluate_models(
        X_train_fn, X_test_fn, y_train_fn, y_test_fn,
        class_names=["Credible / Real", "Unreliable / Fake"],
        task_name="Fake News Detection"
    )

    # Save Best Fake News Model
    best_fn_model = {
        "vectorizer": fake_eval["best_pipeline"]["vectorizer"],
        "classifier": fake_eval["best_pipeline"]["classifier"],
        "model_name": fake_eval["best_model_name"],
        "classes": ["REAL", "FAKE"]
    }
    joblib.dump(best_fn_model, os.path.join(MODELS_DIR, "fake_news_model.joblib"))
    print(f"[SAVED] Fake News Model ({fake_eval['best_model_name']}) -> models/fake_news_model.joblib")

    # 2. Clickbait Dataset & Training
    df_cb = prepare_clickbait_data()
    X_train_cb, X_test_cb, y_train_cb, y_test_cb = train_test_split(
        df_cb["headline"],
        df_cb["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_cb["label"]
    )

    cb_eval = evaluate_models(
        X_train_cb, X_test_cb, y_train_cb, y_test_cb,
        class_names=["Non-Clickbait", "Clickbait"],
        task_name="Clickbait Detection"
    )

    # Save Best Clickbait Model
    best_cb_model = {
        "vectorizer": cb_eval["best_pipeline"]["vectorizer"],
        "classifier": cb_eval["best_pipeline"]["classifier"],
        "model_name": cb_eval["best_model_name"],
        "classes": ["Non-Clickbait", "Clickbait"]
    }
    joblib.dump(best_cb_model, os.path.join(MODELS_DIR, "clickbait_model.joblib"))
    print(f"[SAVED] Clickbait Model ({cb_eval['best_model_name']}) -> models/clickbait_model.joblib")

    # 3. Headline-Body Mismatch Detector
    mismatch_pipeline, mismatch_stats = train_and_evaluate_mismatch(df_fake)
    joblib.dump(mismatch_pipeline, os.path.join(MODELS_DIR, "mismatch_detector.joblib"))
    print("[SAVED] Mismatch Detector -> models/mismatch_detector.joblib")

    # 4. Compile Comprehensive Evaluation Report
    full_evaluation = {
        "metadata": {
            "project_name": "NewsGuard: Explainable Fake News, Clickbait & Headline-Body Mismatch Detection",
            "evaluated_at": timestamp,
            "random_seed": RANDOM_STATE,
            "split_ratio": "80% Train, 20% Held-Out Test (Stratified)"
        },
        "fake_news_task": {
            "dataset_name": "Fake or Real News Dataset (George McIntire / Lutz Hamel)",
            "dataset_url": "https://github.com/lutzhamel/fake-news",
            "total_samples": len(df_fake),
            "train_samples": len(X_train_fn),
            "test_samples": len(X_test_fn),
            "class_distribution": df_fake["label"].value_counts().to_dict(),
            "best_model": fake_eval["best_model_name"],
            "models_comparison": fake_eval["models"],
            "top_features": fake_eval["top_features"]
        },
        "clickbait_task": {
            "dataset_name": "Chakraborty et al. Stop Clickbait Dataset (ASONAM 2016)",
            "dataset_url": "https://github.com/bhargaviparanjape/clickbait",
            "citation": "Chakraborty et al., Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media, IEEE/ACM ASONAM 2016",
            "total_samples": len(df_cb),
            "train_samples": len(X_train_cb),
            "test_samples": len(X_test_cb),
            "class_distribution": df_cb["label_name"].value_counts().to_dict(),
            "best_model": cb_eval["best_model_name"],
            "models_comparison": cb_eval["models"],
            "top_features": cb_eval["top_features"]
        },
        "headline_body_mismatch_task": {
            "methodology": "TF-IDF N-gram Vectorization with Cosine Similarity & Validated Thresholding",
            "metrics": mismatch_stats
        }
    }

    eval_json_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(full_evaluation, f, indent=2)

    print(f"\n[DONE] Full evaluation results exported to: {eval_json_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
