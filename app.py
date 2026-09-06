"""
IDGuardian scoring web app.

Loads all 10 trained models (see train_models.py / models/) once at startup and
exposes a single-page UI + JSON API for scoring one profile at a time through the
full four-layer pipeline: Lexical -> Semantic -> Behavioral -> Trust Fusion.

The UI takes one input -- a demo username or a profile URL -- looks the profile
up in the synthetic demo dataset (see generate_demo_dataset.py), shows exactly
what was fetched, and explains *why* each layer scored it the way it did using
the model's own math (TF-IDF/LogReg word contributions, XGBoost's per-prediction
SHAP-style feature contributions, the Trust Fusion layer's learned weights) --
no external LLM call, so there's nothing to hallucinate a reason that isn't
actually what the model computed.

Run: python app.py   (serves http://127.0.0.1:5000)
"""
from __future__ import annotations

import json
import os
import random
import re
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
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
with open(os.path.join(MODELS_DIR, "behavioral_reference_stats.json")) as f:
    reference_stats = json.load(f)  # see compute_reference_stats.py
print("Models loaded. Ready.")

BOOL_FIELDS = {"is_verified", "has_website", "has_location"}
PLATFORM_TEXT_LABEL = {"instagram": "Bio", "facebook": "About / Bio",
                       "linkedin": "Headline", "twitter": "Bio"}

# Simple, widely-used monochrome brand marks (the same style shipped in icon
# libraries like Simple Icons / Font Awesome) for platform identification in
# the UI -- not official brand assets, just recognizable glyphs.
PLATFORM_ICONS = {
    "instagram": (
        '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="2.5" y="2.5" width="19" height="19" rx="6" stroke="white" stroke-width="2"/>'
        '<circle cx="12" cy="12" r="5" stroke="white" stroke-width="2"/>'
        '<circle cx="17.3" cy="6.7" r="1.15" fill="white"/></svg>'
    ),
    "facebook": (
        '<svg viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M13.5 21v-8.2h2.75l.41-3.19h-3.16V7.55c0-.92.26-1.55 1.58-1.55h1.68V3.14'
        'C15.94 3.06 15.07 3 14.03 3c-2.19 0-3.69 1.34-3.69 3.79v2.82H7.6v3.19h2.74V21h3.16Z"/></svg>'
    ),
    "linkedin": (
        '<svg viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M6.94 8.5H3.56V20h3.38V8.5ZM5.25 3a1.96 1.96 0 1 0 0 3.92A1.96 1.96 0 0 0 5.25 3Z"/>'
        '<path d="M20.45 20h-3.37v-5.6c0-1.34-.02-3.06-1.87-3.06-1.87 0-2.16 1.46-2.16 2.96V20H9.68V8.5h3.24v1.57h.05'
        'c.45-.85 1.55-1.75 3.2-1.75 3.42 0 4.05 2.25 4.05 5.18V20Z"/></svg>'
    ),
    "twitter": (
        '<svg viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M18.9 3H21l-6.44 7.36L22 21h-6.06l-4.75-6.2L5.7 21H3.58l6.83-7.8L3 3h6.2l4.3 5.68L18.9 3Zm-1.06 16.17'
        'h1.14L7.28 4.77H6.05l11.79 14.4Z"/></svg>'
    ),
}
FIELD_LABELS = {
    "followers_count": "Followers", "following_count": "Following",
    "posts_count": "Posts count", "tweets_count": "Tweets count",
    "account_age_days": "Account age (days)",
    "avg_likes": "Avg likes / post", "avg_comments": "Avg comments / post",
    "avg_shares": "Avg shares / post", "profile_completion_score": "Profile completion",
    "friends_count": "Friends", "mutual_friends_count": "Mutual friends",
    "connections_count": "Connections", "endorsements_count": "Endorsements",
    "skills_count": "Skills listed", "is_verified": "Verified", "has_website": "Has website",
    "has_location": "Has location", "engagement_rate": "Engagement rate",
    "follow_ratio": "Follow ratio", "follower_gap": "Follower gap",
    "friend_follower_ratio": "Friend/follower ratio", "mutual_friend_ratio": "Mutual friend ratio",
    "endorsements_per_skill": "Endorsements per skill",
}


def field_label(name: str) -> str:
    return FIELD_LABELS.get(name, name.replace("_", " ").capitalize())


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
SAMPLES_PER_LABEL = 2  # 2 genuine + 2 fake per platform, to fill out the "try examples" row
for p in T.PLATFORMS:
    prof = demo_profiles[p]
    for is_fake_val in (0, 1):
        subset = prof[prof["is_fake"] == is_fake_val]
        if len(subset):
            n = min(SAMPLES_PER_LABEL, len(subset))
            picks = subset.sample(n, random_state=_rng.randint(0, 10_000))
            for _, pick in picks.iterrows():
                SAMPLE_USERNAMES.append({"username": pick["username"], "platform": p,
                                          "is_fake": bool(is_fake_val)})
print(f"Demo dataset loaded: {sum(len(v) for v in demo_profiles.values())} profiles across "
      f"{len(T.PLATFORMS)} platforms.")


# ---------------------------------------------------------------------------
# Single-input parsing: a demo username, or a platform profile URL
# ---------------------------------------------------------------------------
DOMAIN_TO_PLATFORM = {
    "instagram.com": "instagram", "www.instagram.com": "instagram",
    "facebook.com": "facebook", "www.facebook.com": "facebook", "fb.com": "facebook",
    "linkedin.com": "linkedin", "www.linkedin.com": "linkedin",
    "twitter.com": "twitter", "www.twitter.com": "twitter", "x.com": "twitter",
}


def parse_profile_input(raw: str) -> tuple[str | None, str]:
    """Return (platform_hint_or_None, username). Accepts a bare username or a
    profile URL for any of the four platforms; only used to narrow the search
    to one platform when a URL makes that unambiguous -- lookup still falls
    back to a cross-platform username search either way."""
    raw = raw.strip()
    if "://" not in raw and "." in raw.split("/")[0]:
        raw = "https://" + raw  # allow "instagram.com/foo" without a scheme
    if "://" in raw:
        parsed = urlparse(raw)
        platform = DOMAIN_TO_PLATFORM.get(parsed.netloc.lower())
        path_parts = [seg for seg in parsed.path.split("/") if seg]
        if platform == "linkedin" and path_parts and path_parts[0] in ("in", "company"):
            path_parts = path_parts[1:]
        username = path_parts[0] if path_parts else ""
        return platform, username
    return None, raw


# ---------------------------------------------------------------------------
# Per-layer, data-driven explanations -- grounded in the model's own math
# (TF-IDF x LogReg coefficients, XGBoost's per-prediction SHAP-style feature
# contributions, the Trust Fusion layer's learned weights). No LLM call: an
# explanation here is never anything the model didn't actually compute.
# ---------------------------------------------------------------------------

def explain_lexical(pooled_text: str, X_lex, lexical_score: float) -> dict:
    feature_names = tfidf.get_feature_names_out()
    coefs = lexical_model.coef_[0]
    coo = X_lex.tocoo()
    contributions = sorted(
        ((feature_names[j], float(v) * float(coefs[j])) for j, v in zip(coo.col, coo.data)),
        key=lambda x: x[1],
    )
    fake_words = [w for w, c in reversed(contributions) if c > 0][:5]
    genuine_words = [w for w, c in contributions if c < 0][:5]

    if fake_words and genuine_words:
        summary = (f'Words like "{", ".join(fake_words[:3])}" pushed this toward fake, while '
                    f'"{", ".join(genuine_words[:3])}" read as more genuine — net score {lexical_score:.0%}.')
    elif fake_words:
        summary = (f'Words like "{", ".join(fake_words[:3])}" are strongly associated with fake '
                    f'profiles in the training data, with little offsetting genuine-sounding language.')
    elif genuine_words:
        summary = (f'Words like "{", ".join(genuine_words[:3])}" are strongly associated with genuine '
                    f'profiles in the training data, with little fake-sounding language to offset them.')
    else:
        summary = ("No strongly distinctive words were found in this text (it may be short or generic) — "
                    "the score reflects a fairly neutral vocabulary.")
    return {"top_fake_words": fake_words, "top_genuine_words": genuine_words, "summary": summary}


def explain_semantic(semantic_score: float, lexical_score: float) -> dict:
    tone = "fake-leaning" if semantic_score >= 0.5 else "genuine-leaning"
    agree = (semantic_score >= 0.5) == (lexical_score >= 0.5)
    agreement_text = "agreeing with" if agree else "diverging from"
    summary = (f"The semantic layer reads overall tone, phrasing, and context using sentence "
               f"embeddings — not individual keywords — and found this profile's writing style to be "
               f"{tone} ({semantic_score:.0%}), {agreement_text} the lexical layer's word-based read.")
    return {"summary": summary}


def explain_behavioral(row: dict, cols: list[str], platform: str, X_beh: np.ndarray) -> dict:
    booster = behavioral_models[platform].get_booster()
    dm = xgb.DMatrix(X_beh, feature_names=cols)
    contribs = booster.predict(dm, pred_contribs=True)[0][:-1]  # drop trailing bias term
    ranked = sorted(zip(cols, contribs), key=lambda x: -abs(x[1]))[:4]

    ref = reference_stats.get(platform, {})
    details = []
    for name, contrib in ranked:
        r = ref.get(name, {})
        details.append({
            "feature": field_label(name), "value": round(float(row[name]), 3),
            "genuine_typical": r.get("genuine_mean"), "fake_typical": r.get("fake_mean"),
            "direction": "fake" if contrib > 0 else "genuine",
            "contribution": round(float(contrib), 3),
        })

    if details:
        pieces = []
        for d in details[:3]:
            g, fk = d["genuine_typical"], d["fake_typical"]
            ref_txt = f" (genuine avg {g:g}, fake avg {fk:g})" if g is not None and fk is not None else ""
            pieces.append(f"{d['feature'].lower()}={d['value']:g}{ref_txt} pointed toward {d['direction'].upper()}")
        summary = "Most influential signals: " + "; ".join(pieces) + "."
    else:
        summary = "No single feature stood out strongly; the score reflects a broad combination of signals."
    return {"top_features": details, "summary": summary}


def explain_fusion(platform: str, lexical_score: float, semantic_score: float,
                    behavioral_score: float, fusion_score: float) -> dict:
    weights = fusion_models[platform].coef_[0]
    names = ["Lexical", "Semantic", "Behavioral"]
    scores = [lexical_score, semantic_score, behavioral_score]
    ranked = sorted(zip(names, weights, scores), key=lambda x: -abs(x[1]))
    top_name = ranked[0][0]
    summary = (f"Trust Fusion learned how much to trust each layer from validation data; for "
               f"{platform.capitalize()}, {top_name} carries the most weight. The three layers scored "
               f"{scores[0]:.0%} (Lexical), {scores[1]:.0%} (Semantic), and {scores[2]:.0%} (Behavioral), "
               f"combining into a final {fusion_score:.0%}.")
    return {"weights": {n: round(float(w), 3) for n, w, _ in ranked}, "summary": summary}


def score_profile(platform: str, text: str, captions: list[str], hashtags: list[str],
                   behavioral_raw: dict) -> dict:
    """Run one profile through all four layers and attach a data-driven
    explanation for each. `behavioral_raw` maps raw column name -> value
    (bools as bool/0/1, everything else numeric); derived ratios are computed
    here the same way train_models.py computes them for training."""
    captions = list(captions)
    hashtags = list(hashtags)
    pooled_text = " ".join([text] + captions[:T.N_CAPTIONS_FOR_TEXT] + hashtags).strip()
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
        "explanations": {
            "lexical": explain_lexical(pooled_text, X_lex, lexical_score),
            "semantic": explain_semantic(semantic_score, lexical_score),
            "behavioral": explain_behavioral(row, cols, platform, X_beh),
            "fusion": explain_fusion(platform, lexical_score, semantic_score, behavioral_score, fusion_score),
        },
        "profile_data": {
            "text_label": PLATFORM_TEXT_LABEL[platform],
            "text": text,
            "captions": captions[:T.N_CAPTIONS_FOR_TEXT],
            "hashtags": hashtags,
            "behavioral": [
                {"label": field_label(c), "value": row[c], "is_bool": c in BOOL_FIELDS}
                for c in T.BEHAVIORAL_BASE_COLS[platform]
            ],
        },
    }


def platform_field_spec(platform: str) -> list[dict]:
    fields = []
    for col in T.BEHAVIORAL_BASE_COLS[platform]:
        fields.append({
            "name": col,
            "label": field_label(col),
            "type": "bool" if col in BOOL_FIELDS else
                    ("float" if col == "profile_completion_score" else "number"),
        })
    return fields


@app.route("/")
def index():
    specs = {p: platform_field_spec(p) for p in T.PLATFORMS}
    return render_template("index.html", platforms=T.PLATFORMS, specs=specs,
                           text_labels=PLATFORM_TEXT_LABEL, sample_usernames=SAMPLE_USERNAMES,
                           platform_icons=PLATFORM_ICONS)


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
    query = (request.args.get("query") or request.args.get("username") or "").strip()
    if not query:
        return jsonify({"error": "a username or profile URL is required"}), 400

    platform_hint, username = parse_profile_input(query)
    username = username.strip().lower()
    if not username:
        return jsonify({"error": "couldn't find a username in that input"}), 400

    match = username_index.get(username)
    if not match:
        return jsonify({"error": f"No demo profile found for '{username}'. "
                                  f"Try one of the sample usernames below."}), 404
    platform, user_id = match
    if platform_hint and platform_hint != platform:
        return jsonify({"error": f"'{username}' exists in the demo set, but on "
                                  f"{platform.capitalize()}, not {platform_hint.capitalize()}."}), 404

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
