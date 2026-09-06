"""
IDGuardian scoring web app.

Loads all 10 trained models (see train_models.py / models/) once at startup and
exposes a single-page UI + JSON API for scoring one profile at a time through the
full four-layer pipeline: Lexical -> Semantic -> Behavioral -> Trust Fusion.

Run: python app.py   (serves http://127.0.0.1:5000)
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import train_models as T

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = T.MODELS_DIR

app = Flask(__name__)

print("Loading models...")
tfidf = joblib.load(os.path.join(MODELS_DIR, "lexical_tfidf_vectorizer.joblib"))
lexical_model = joblib.load(os.path.join(MODELS_DIR, "lexical_logreg.joblib"))
semantic_model = joblib.load(os.path.join(MODELS_DIR, "semantic_xgboost.joblib"))

with open(os.path.join(MODELS_DIR, "semantic_sbert_model.txt")) as f:
    SBERT_MODEL_NAME = f.read().strip()

from sentence_transformers import SentenceTransformer  # noqa: E402  (heavy import, after Flask setup)
sbert = SentenceTransformer(SBERT_MODEL_NAME)

behavioral_models = {p: joblib.load(os.path.join(MODELS_DIR, f"behavioral_{p}_xgboost.joblib"))
                     for p in T.PLATFORMS}
fusion_models = {p: joblib.load(os.path.join(MODELS_DIR, f"fusion_{p}_logreg.joblib"))
                 for p in T.PLATFORMS}
with open(os.path.join(MODELS_DIR, "platform_behavioral_columns.json")) as f:
    behavioral_columns = json.load(f)
print("Models loaded. Ready.")

# Field metadata the frontend uses to render each platform's behavioral form and
# labels -- kept here (not just in train_models.py) since it's presentation info,
# not modeling logic.
FIELD_LABELS = {
    "followers_count": "Followers", "following_count": "Following",
    "posts_count": "Posts count", "tweets_count": "Tweets count",
    "account_age_days": "Account age (days)",
    "avg_likes": "Avg likes / post", "avg_comments": "Avg comments / post",
    "avg_shares": "Avg shares / post", "profile_completion_score": "Profile completion (0-1)",
    "friends_count": "Friends", "mutual_friends_count": "Mutual friends",
    "connections_count": "Connections", "endorsements_count": "Endorsements",
    "skills_count": "Skills listed",
}
BOOL_FIELDS = {"is_verified", "has_website", "has_location"}
PLATFORM_TEXT_LABEL = {"instagram": "Bio", "facebook": "About / Bio",
                       "linkedin": "Headline", "twitter": "Bio"}


def platform_field_spec(platform: str) -> list[dict]:
    fields = []
    for col in T.BEHAVIORAL_BASE_COLS[platform]:
        fields.append({
            "name": col,
            "label": FIELD_LABELS.get(col, col.replace("_", " ").title()),
            "type": "bool" if col in BOOL_FIELDS else
                    ("float" if col == "profile_completion_score" else "number"),
        })
    return fields


@app.route("/")
def index():
    specs = {p: platform_field_spec(p) for p in T.PLATFORMS}
    return render_template("index.html", platforms=T.PLATFORMS, specs=specs,
                           text_labels=PLATFORM_TEXT_LABEL)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    platform = data.get("platform")
    if platform not in T.PLATFORMS:
        return jsonify({"error": f"platform must be one of {T.PLATFORMS}"}), 400

    text = (data.get("text") or "").strip()
    captions = [c for c in (data.get("captions") or []) if c.strip()]
    hashtags = [h for h in (data.get("hashtags") or []) if h.strip()]
    pooled_text = " ".join([text] + captions[:T.N_CAPTIONS_FOR_TEXT] + hashtags).strip()
    if not pooled_text:
        pooled_text = "no bio provided"

    # ---- Lexical ----
    X_lex = tfidf.transform([pooled_text])
    lexical_score = float(lexical_model.predict_proba(X_lex)[0, 1])

    # ---- Semantic ----
    emb = sbert.encode([pooled_text])
    semantic_score = float(semantic_model.predict_proba(emb)[0, 1])

    # ---- Behavioral ----
    raw = data.get("behavioral") or {}
    row = {}
    for c in T.BEHAVIORAL_BASE_COLS[platform]:
        v = raw.get(c, 0)
        row[c] = float(bool(v)) if c in BOOL_FIELDS else float(v or 0)
    derived = T.DERIVED_COLS[platform](pd.DataFrame([row]))
    for k, v in derived.items():
        row[k] = float(v.iloc[0])
    cols = behavioral_columns[platform]
    X_beh = np.array([[row[c] for c in cols]], dtype=float)
    behavioral_score = float(behavioral_models[platform].predict_proba(X_beh)[0, 1])

    # ---- Trust Fusion ----
    X_fusion = np.array([[lexical_score, semantic_score, behavioral_score]])
    fusion_score = float(fusion_models[platform].predict_proba(X_fusion)[0, 1])

    return jsonify({
        "platform": platform,
        "lexical_score": round(lexical_score, 4),
        "semantic_score": round(semantic_score, 4),
        "behavioral_score": round(behavioral_score, 4),
        "fusion_score": round(fusion_score, 4),
        "verdict": "Likely Fake" if fusion_score >= 0.5 else "Likely Genuine",
    })


if __name__ == "__main__":
    # debug/reloader off on purpose -- the reloader re-imports this module in a
    # child process, which means loading SBERT + every joblib model twice.
    app.run(debug=False, port=5000)
