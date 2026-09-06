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
import random

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

# ---------------------------------------------------------------------------
# Demo lookup dataset (see generate_demo_dataset.py) -- 200 profiles/platform
# the models have never seen in training or testing, so a username lookup is
# an honest generalization check rather than replaying a memorized example.
# ---------------------------------------------------------------------------
print("Loading demo lookup dataset...")
demo_profiles: dict[str, pd.DataFrame] = {}
demo_posts: dict[str, pd.DataFrame] = {}
username_index: dict[str, tuple[str, str]] = {}  # lowercase username -> (platform, user_id)

for p in T.PLATFORMS:
    prof = pd.read_csv(os.path.join(T.DATA_DIR, f"demo_{p}_profiles.csv"))
    posts = pd.read_csv(os.path.join(T.DATA_DIR, f"demo_{p}_posts.csv"), keep_default_na=False)
    demo_profiles[p] = prof.set_index("user_id", drop=False)
    demo_posts[p] = posts
    for _, row in prof.iterrows():
        username_index.setdefault(row["username"].lower(), (p, row["user_id"]))

_rng = random.Random(7)
SAMPLE_USERNAMES = []
for p in T.PLATFORMS:
    prof = demo_profiles[p]
    for is_fake_val in (0, 1):
        subset = prof[prof["is_fake"] == is_fake_val]
        if len(subset):
            pick = subset.sample(1, random_state=_rng.randint(0, 10_000)).iloc[0]
            SAMPLE_USERNAMES.append({"username": pick["username"], "platform": p,
                                      "is_fake": bool(is_fake_val)})
print(f"Demo dataset loaded: {sum(len(v) for v in demo_profiles.values())} profiles across "
      f"{len(T.PLATFORMS)} platforms.")


def score_profile(platform: str, text: str, captions: list[str], hashtags: list[str],
                   behavioral_raw: dict) -> dict:
    """Run one profile through all four layers. `behavioral_raw` maps raw column
    name -> value (bools as bool/0/1, everything else numeric); derived ratios are
    computed here the same way train_models.py computes them for training."""
    pooled_text = " ".join([text] + list(captions[:T.N_CAPTIONS_FOR_TEXT]) + list(hashtags)).strip()
    if not pooled_text:
        pooled_text = "no bio provided"

    X_lex = tfidf.transform([pooled_text])
    lexical_score = float(lexical_model.predict_proba(X_lex)[0, 1])

    emb = sbert.encode([pooled_text])
    semantic_score = float(semantic_model.predict_proba(emb)[0, 1])

    row = {}
    for c in T.BEHAVIORAL_BASE_COLS[platform]:
        v = behavioral_raw.get(c, 0)
        row[c] = float(bool(v)) if c in BOOL_FIELDS else float(v or 0)
    derived = T.DERIVED_COLS[platform](pd.DataFrame([row]))
    for k, v in derived.items():
        row[k] = float(v.iloc[0])
    cols = behavioral_columns[platform]
    X_beh = np.array([[row[c] for c in cols]], dtype=float)
    behavioral_score = float(behavioral_models[platform].predict_proba(X_beh)[0, 1])

    X_fusion = np.array([[lexical_score, semantic_score, behavioral_score]])
    fusion_score = float(fusion_models[platform].predict_proba(X_fusion)[0, 1])

    return {
        "platform": platform,
        "lexical_score": round(lexical_score, 4),
        "semantic_score": round(semantic_score, 4),
        "behavioral_score": round(behavioral_score, 4),
        "fusion_score": round(fusion_score, 4),
        "verdict": "Likely Fake" if fusion_score >= 0.5 else "Likely Genuine",
    }

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
                           text_labels=PLATFORM_TEXT_LABEL, sample_usernames=SAMPLE_USERNAMES)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    platform = data.get("platform")
    if platform not in T.PLATFORMS:
        return jsonify({"error": f"platform must be one of {T.PLATFORMS}"}), 400

    text = (data.get("text") or "").strip()
    captions = [c for c in (data.get("captions") or []) if c.strip()]
    hashtags = [h for h in (data.get("hashtags") or []) if h.strip()]
    result = score_profile(platform, text, captions, hashtags, data.get("behavioral") or {})
    return jsonify(result)


@app.route("/api/lookup", methods=["GET"])
def lookup():
    username = (request.args.get("username") or "").strip().lower()
    if not username:
        return jsonify({"error": "username is required"}), 400

    match = username_index.get(username)
    if not match:
        return jsonify({"error": f"No demo profile found for username '{username}'. "
                                  f"Try one of the sample usernames below."}), 404
    platform, user_id = match

    row = demo_profiles[platform].loc[user_id]
    posts = demo_posts[platform]
    user_posts = posts[posts["user_id"] == user_id]

    text = row[T.TEXT_COL[platform]]
    captions = user_posts["caption"].tolist()
    hashtags: list[str] = []
    for h in user_posts["hashtags"].tolist():
        if h:
            hashtags.extend(h.split())

    behavioral_raw = {c: row[c] for c in T.BEHAVIORAL_BASE_COLS[platform]}
    result = score_profile(platform, text, captions, hashtags, behavioral_raw)
    result["username"] = str(row["username"])
    result["ground_truth_archetype"] = str(row["archetype"])
    result["ground_truth_is_fake"] = bool(row["is_fake"])
    return jsonify(result)


if __name__ == "__main__":
    # debug/reloader off on purpose -- the reloader re-imports this module in a
    # child process, which means loading SBERT + every joblib model twice.
    app.run(debug=False, port=5000)
