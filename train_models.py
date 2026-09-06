"""
IDGuardian four-layer model training pipeline.

Implements the architecture from DATASET_GENERATION_SPEC.md Section 8:
  - Lexical Layer   (TF-IDF + Logistic Regression) -- ONE shared model, pooled text
                    (bio/headline + captions + hashtags) across all four platforms.
  - Semantic Layer  (Sentence-BERT + XGBoost)       -- ONE shared model, pooled SBERT
                    embeddings across all four platforms.
  - Behavioral Layer (XGBoost)                      -- FOUR per-platform models, on each
                    platform's own numeric/behavioral feature set.
  - Trust Fusion Layer (Logistic Regression)         -- FOUR per-platform models, combining
                    that platform's lexical/semantic/behavioral scores into one probability.

Net: 2 shared + 8 platform-specific = 10 trained models, matching the spec exactly.

Methodology (why this isn't just "fit everything on everything"):
  Each platform's profiles are split 70/30 (stratified on is_fake, seed=42) into train/test.
  All four platforms' train profiles are pooled to fit the two shared layers; the held-out
  30% test profiles are NEVER used for fitting anything. The Trust Fusion layer is trained on
  each base layer's *out-of-fold* predictions on the training set (5-fold stratified CV) --
  feeding a layer its own in-sample predictions would let the fusion model see answers that
  came from a model that already memorized them, understating how much fusion actually adds
  over the base layers. Final per-layer models are refit on the *full* training set (out-of-
  fold refitting is only for generating honest fusion-training features) and are what actually
  score the untouched test set for the reported metrics and what gets saved to models/.

Usage: python train_models.py
"""
from __future__ import annotations

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SEED = 42
TEST_SIZE = 0.30
N_FOLDS = 5
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
N_CAPTIONS_FOR_TEXT = 8   # cap how many of a profile's captions feed the text layers

PLATFORMS = ["instagram", "facebook", "linkedin", "twitter"]

TEXT_COL = {"instagram": "bio", "facebook": "bio", "linkedin": "headline", "twitter": "bio"}
POSTS_COUNT_COL = {"instagram": "posts_count", "facebook": "posts_count",
                    "linkedin": "posts_count", "twitter": "tweets_count"}

# Numeric/boolean behavioral columns per platform -- "the same kind of signals a real
# system would see" (spec Rule 6), including the documented-but-not-stored derived ratios.
BEHAVIORAL_BASE_COLS = {
    "instagram": ["followers_count", "following_count", "posts_count", "account_age_days",
                  "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                  "profile_completion_score", "is_verified", "has_website", "has_location"],
    "twitter": ["followers_count", "following_count", "tweets_count", "account_age_days",
                "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                "profile_completion_score", "is_verified", "has_website", "has_location"],
    "facebook": ["friends_count", "followers_count", "posts_count", "account_age_days",
                 "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                 "mutual_friends_count", "profile_completion_score",
                 "is_verified", "has_website", "has_location"],
    "linkedin": ["connections_count", "posts_count", "account_age_days",
                 "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                 "endorsements_count", "skills_count", "profile_completion_score",
                 "is_verified", "has_website", "has_location"],
}
# Derived columns computed here (not stored in the CSVs -- see NOTES.md) and appended
# to the base columns above for behavioral-layer training.
DERIVED_COLS = {
    "instagram": lambda df: {"follow_ratio": df["followers_count"] / (df["following_count"] + 1),
                              "follower_gap": df["followers_count"] - df["following_count"]},
    "twitter": lambda df: {"follow_ratio": df["followers_count"] / (df["following_count"] + 1),
                            "follower_gap": df["followers_count"] - df["following_count"]},
    "facebook": lambda df: {"friend_follower_ratio": df["friends_count"] / (df["followers_count"] + 1),
                             "mutual_friend_ratio": df["mutual_friends_count"] / (df["friends_count"] + 1)},
    "linkedin": lambda df: {"endorsements_per_skill": df["endorsements_count"] / (df["skills_count"] + 1)},
}


def behavioral_columns(platform: str) -> list[str]:
    return BEHAVIORAL_BASE_COLS[platform] + list(DERIVED_COLS[platform](
        pd.DataFrame({c: [1] for c in BEHAVIORAL_BASE_COLS[platform]})
    ).keys())


def load_platform(platform: str) -> pd.DataFrame:
    """Load one platform's profiles+posts and return a profiles df enriched with:
    - `pooled_text`: bio/headline + up to N_CAPTIONS_FOR_TEXT captions + all hashtags
    - derived behavioral ratio columns
    - a `platform` column
    """
    profiles = pd.read_csv(os.path.join(DATA_DIR, f"{platform}_profiles.csv"))
    posts = pd.read_csv(os.path.join(DATA_DIR, f"{platform}_posts.csv"), keep_default_na=False)

    posts_sorted = posts.sort_values("user_id")
    grouped_captions = posts_sorted.groupby("user_id")["caption"].apply(
        lambda s: " ".join(s.iloc[:N_CAPTIONS_FOR_TEXT])
    )
    grouped_hashtags = posts_sorted.groupby("user_id")["hashtags"].apply(
        lambda s: " ".join(t for t in s if t)
    )

    profiles = profiles.set_index("user_id")
    text_col = TEXT_COL[platform]
    profile_text = profiles[text_col].fillna("")
    captions_text = grouped_captions.reindex(profiles.index).fillna("")
    hashtags_text = grouped_hashtags.reindex(profiles.index).fillna("")
    profiles["pooled_text"] = (profile_text + " " + captions_text + " " + hashtags_text).str.strip()
    profiles = profiles.reset_index()

    for name, series in DERIVED_COLS[platform](profiles).items():
        profiles[name] = series

    for col in ["is_verified", "has_website", "has_location"]:
        profiles[col] = profiles[col].astype(int)

    profiles["platform"] = platform
    return profiles


def metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Loading platform data and building pooled text...")
    platform_dfs = {p: load_platform(p) for p in PLATFORMS}

    # ---- 1. Per-platform train/test split (stratified on is_fake) ----
    train_parts, test_parts = [], []
    for p in PLATFORMS:
        df = platform_dfs[p]
        tr, te = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["is_fake"])
        tr = tr.reset_index(drop=True)
        te = te.reset_index(drop=True)
        train_parts.append(tr)
        test_parts.append(te)
    train_all = pd.concat(train_parts, ignore_index=True)
    test_all = pd.concat(test_parts, ignore_index=True)
    print(f"  train={len(train_all)}  test={len(test_all)}  (per-platform 70/30 stratified split)")

    y_train_all = train_all["is_fake"].to_numpy()
    y_test_all = test_all["is_fake"].to_numpy()

    report = {"platforms": {}, "shared": {}}

    # =======================================================================
    # LEXICAL LAYER (shared): TF-IDF + Logistic Regression
    # =======================================================================
    print("Training lexical layer (TF-IDF + Logistic Regression, shared)...")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")
    X_lex_train = tfidf.fit_transform(train_all["pooled_text"])
    X_lex_test = tfidf.transform(test_all["pooled_text"])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lex_oof = np.zeros(len(train_all))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_lex_train, y_train_all)):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_lex_train[tr_idx], y_train_all[tr_idx])
        lex_oof[val_idx] = clf.predict_proba(X_lex_train[val_idx])[:, 1]

    lexical_model = LogisticRegression(max_iter=2000)
    lexical_model.fit(X_lex_train, y_train_all)
    lex_test_proba = lexical_model.predict_proba(X_lex_test)[:, 1]
    lex_test_pred = (lex_test_proba >= 0.5).astype(int)

    report["shared"]["lexical_overall"] = metrics(y_test_all, lex_test_pred, lex_test_proba)
    print(f"  lexical overall (pooled test): {report['shared']['lexical_overall']}")

    joblib.dump(tfidf, os.path.join(MODELS_DIR, "lexical_tfidf_vectorizer.joblib"))
    joblib.dump(lexical_model, os.path.join(MODELS_DIR, "lexical_logreg.joblib"))

    # =======================================================================
    # SEMANTIC LAYER (shared): Sentence-BERT + XGBoost
    # =======================================================================
    print(f"Encoding pooled text with SBERT ({SBERT_MODEL_NAME})... (this takes a few minutes on CPU)")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(SBERT_MODEL_NAME)
    emb_train = sbert.encode(train_all["pooled_text"].tolist(), batch_size=64, show_progress_bar=True)
    emb_test = sbert.encode(test_all["pooled_text"].tolist(), batch_size=64, show_progress_bar=True)

    print("Training semantic layer (XGBoost on SBERT embeddings, shared)...")
    sem_oof = np.zeros(len(train_all))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(emb_train, y_train_all)):
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                 random_state=SEED, eval_metric="logloss")
        clf.fit(emb_train[tr_idx], y_train_all[tr_idx])
        sem_oof[val_idx] = clf.predict_proba(emb_train[val_idx])[:, 1]

    semantic_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                        random_state=SEED, eval_metric="logloss")
    semantic_model.fit(emb_train, y_train_all)
    sem_test_proba = semantic_model.predict_proba(emb_test)[:, 1]
    sem_test_pred = (sem_test_proba >= 0.5).astype(int)

    report["shared"]["semantic_overall"] = metrics(y_test_all, sem_test_pred, sem_test_proba)
    print(f"  semantic overall (pooled test): {report['shared']['semantic_overall']}")

    joblib.dump(semantic_model, os.path.join(MODELS_DIR, "semantic_xgboost.joblib"))
    with open(os.path.join(MODELS_DIR, "semantic_sbert_model.txt"), "w") as f:
        f.write(SBERT_MODEL_NAME + "\n")

    train_all = train_all.copy()
    test_all = test_all.copy()
    train_all["lexical_oof"] = lex_oof
    train_all["semantic_oof"] = sem_oof
    test_all["lexical_score"] = lex_test_proba
    test_all["semantic_score"] = sem_test_proba

    # =======================================================================
    # BEHAVIORAL LAYER (per platform): XGBoost
    # + TRUST FUSION LAYER (per platform): Logistic Regression on [lex, sem, beh]
    # =======================================================================
    for p in PLATFORMS:
        print(f"Training behavioral + fusion layers for {p}...")
        cols = behavioral_columns(p)
        tr_p = train_all[train_all["platform"] == p].reset_index(drop=True)
        te_p = test_all[test_all["platform"] == p].reset_index(drop=True)

        X_beh_train = tr_p[cols].to_numpy(dtype=float)
        y_beh_train = tr_p["is_fake"].to_numpy()
        X_beh_test = te_p[cols].to_numpy(dtype=float)
        y_beh_test = te_p["is_fake"].to_numpy()

        beh_oof = np.zeros(len(tr_p))
        skf_p = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for tr_idx, val_idx in skf_p.split(X_beh_train, y_beh_train):
            clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     random_state=SEED, eval_metric="logloss")
            clf.fit(X_beh_train[tr_idx], y_beh_train[tr_idx])
            beh_oof[val_idx] = clf.predict_proba(X_beh_train[val_idx])[:, 1]

        behavioral_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                              random_state=SEED, eval_metric="logloss")
        behavioral_model.fit(X_beh_train, y_beh_train)
        beh_test_proba = behavioral_model.predict_proba(X_beh_test)[:, 1]
        beh_test_pred = (beh_test_proba >= 0.5).astype(int)

        joblib.dump(behavioral_model, os.path.join(MODELS_DIR, f"behavioral_{p}_xgboost.joblib"))

        # Fusion training features = that platform's OOF predictions from all 3 base layers
        lex_oof_p = tr_p["lexical_oof"].to_numpy()
        sem_oof_p = tr_p["semantic_oof"].to_numpy()
        X_fusion_train = np.column_stack([lex_oof_p, sem_oof_p, beh_oof])
        fusion_model = LogisticRegression(max_iter=1000)
        fusion_model.fit(X_fusion_train, y_beh_train)
        joblib.dump(fusion_model, os.path.join(MODELS_DIR, f"fusion_{p}_logreg.joblib"))

        # Fusion evaluation uses the FINAL base-layer scores on the untouched test set
        lex_test_p = te_p["lexical_score"].to_numpy()
        sem_test_p = te_p["semantic_score"].to_numpy()
        X_fusion_test = np.column_stack([lex_test_p, sem_test_p, beh_test_proba])
        fusion_test_proba = fusion_model.predict_proba(X_fusion_test)[:, 1]
        fusion_test_pred = (fusion_test_proba >= 0.5).astype(int)

        report["platforms"][p] = {
            "n_train": len(tr_p), "n_test": len(te_p),
            "lexical": metrics(y_beh_test, (lex_test_p >= 0.5).astype(int), lex_test_p),
            "semantic": metrics(y_beh_test, (sem_test_p >= 0.5).astype(int), sem_test_p),
            "behavioral": metrics(y_beh_test, beh_test_pred, beh_test_proba),
            "fusion": metrics(y_beh_test, fusion_test_pred, fusion_test_proba),
            "behavioral_columns": cols,
        }
        print(f"  {p}: behavioral={report['platforms'][p]['behavioral']} "
              f"fusion={report['platforms'][p]['fusion']}")

    with open(os.path.join(MODELS_DIR, "platform_behavioral_columns.json"), "w") as f:
        json.dump({p: behavioral_columns(p) for p in PLATFORMS}, f, indent=2)

    with open(os.path.join(DATA_DIR, "model_training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    write_markdown_report(report)
    print("\nDone. Models saved to models/, report saved to data/model_training_report.md")


def write_markdown_report(report: dict):
    lines = []
    lines.append("# IDGuardian Model Training Report\n")
    lines.append(f"10 models trained per DATASET_GENERATION_SPEC.md Section 8: "
                 f"2 shared (Lexical, Semantic) + 8 per-platform (Behavioral x4, Fusion x4).\n")
    lines.append("## Methodology\n")
    lines.append(f"- Per-platform 70/30 train/test split, stratified on `is_fake`, seed={SEED}.")
    lines.append("- Lexical (TF-IDF + LogisticRegression) and Semantic (SBERT + XGBoost) are "
                 "fit once on all four platforms' pooled training text.")
    lines.append("- Behavioral (XGBoost) and Trust Fusion (LogisticRegression) are fit "
                 "separately per platform.")
    lines.append(f"- Trust Fusion is trained on {N_FOLDS}-fold out-of-fold predictions from the "
                 "other three layers (never on their in-sample predictions), so its reported "
                 "lift over any single layer isn't inflated by a layer seeing its own answers.")
    lines.append("- All metrics below are computed on the held-out test set only.\n")

    lines.append("## Shared layers (pooled across all 4 platforms)\n")
    lines.append("| Layer | Accuracy | F1 | ROC-AUC |")
    lines.append("|---|---|---|---|")
    for layer in ["lexical_overall", "semantic_overall"]:
        m = report["shared"][layer]
        lines.append(f"| {layer.replace('_overall', '').capitalize()} | {m['accuracy']} | {m['f1']} | {m['roc_auc']} |")

    lines.append("\n## Per-platform results\n")
    for p, r in report["platforms"].items():
        lines.append(f"### {p.capitalize()} (train={r['n_train']}, test={r['n_test']})\n")
        lines.append("| Layer | Accuracy | F1 | ROC-AUC |")
        lines.append("|---|---|---|---|")
        for layer in ["lexical", "semantic", "behavioral", "fusion"]:
            m = r[layer]
            lines.append(f"| {layer.capitalize()} | {m['accuracy']} | {m['f1']} | {m['roc_auc']} |")
        best_single = max(r["lexical"]["roc_auc"], r["semantic"]["roc_auc"], r["behavioral"]["roc_auc"])
        lift = r["fusion"]["roc_auc"] - best_single
        lines.append(f"\nFusion vs. best single layer (ROC-AUC): {r['fusion']['roc_auc']} vs. "
                     f"{best_single:.4f} ({'+' if lift >= 0 else ''}{lift:.4f})\n")

    with open(os.path.join(DATA_DIR, "model_training_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
