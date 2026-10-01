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

Run (dev or production -- both use the same production-grade waitress WSGI
server; see the __main__ block at the bottom):
    python app.py                          # serves http://127.0.0.1:5000
    HOST=0.0.0.0 PORT=8080 python app.py   # override bind address/port
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import db
import train_models as T
from instagram_scraper import scrape_instagram_profile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = T.MODELS_DIR
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ---------------------------------------------------------------------------
# Logging -- console + a rotating file (5MB x 3 backups), so a deployment
# doesn't need to rely on scraping stdout for anything beyond the immediate
# session. Startup diagnostics below still use print() on purpose (they're
# one-time human-readable status lines at import time, before the logger's
# handlers -- or even a container's log driver -- are necessarily attached);
# request-time errors and rate-limit events go through `logger`.
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("idguardian")
logger.setLevel(logging.INFO)
_file_handler = RotatingFileHandler(os.path.join(LOG_DIR, "app.log"), maxBytes=5_000_000, backupCount=3)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(_file_handler)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(_console_handler)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1MB cap on any request body

# Only trust X-Forwarded-For (for rate-limiting by real client IP, not a
# proxy's IP) when this app is actually deployed behind a reverse proxy/load
# balancer -- blindly trusting it otherwise would let any client spoof their
# apparent IP and dodge rate limits.
if os.environ.get("TRUST_PROXY") == "1":
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

limiter = Limiter(get_remote_address, app=app, default_limits=["200 per hour"], storage_uri="memory://")


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": f"Rate limit exceeded: {e.description}"}), 429


@app.errorhandler(500)
def internal_error_handler(e):
    logger.exception("Unhandled server error")
    return jsonify({"error": "Internal server error"}), 500


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response


print("Loading models...")
tfidf = joblib.load(os.path.join(MODELS_DIR, "lexical_tfidf_vectorizer.joblib"))
lexical_model = joblib.load(os.path.join(MODELS_DIR, "lexical_logreg.joblib"))
semantic_model = joblib.load(os.path.join(MODELS_DIR, "semantic_xgboost.joblib"))

with open(os.path.join(MODELS_DIR, "semantic_sbert_model.txt")) as f:
    SBERT_MODEL_NAME = f.read().strip()

# The Semantic layer's default backend is PyTorch, which on some locked-down
# Windows machines gets blocked at import time by an Application Control/WDAC
# policy (an OSError naming a specific unsigned .dll -- PyPI's torch wheels
# aren't code-signed, so this can recur any time the venv is rebuilt, even
# though it happened to clear on its own last time). ONNX Runtime's DLLs
# *are* Microsoft-signed, so try that backend first; fall back to the default
# (torch) backend if ONNX isn't available for some reason; only if both fail
# does the app degrade (Lexical + Behavioral + Fusion keep working, and the
# UI says plainly that Semantic is unavailable instead of faking a score).
sbert = None
SEMANTIC_AVAILABLE = False
SEMANTIC_BACKEND = None
SEMANTIC_UNAVAILABLE_REASON = ""
try:
    # Even this import can transitively pull in torch (sentence_transformers ->
    # transformers -> torch at module level in some version combos), so it has
    # to be inside the try too -- picking backend="onnx" below doesn't help if
    # the import itself already failed.
    from sentence_transformers import SentenceTransformer
    try:
        sbert = SentenceTransformer(SBERT_MODEL_NAME, backend="onnx",
                                    model_kwargs={"file_name": "onnx/model.onnx"})
        SEMANTIC_BACKEND = "onnx"
    except Exception as onnx_exc:
        print(f"Semantic layer: ONNX backend unavailable ({onnx_exc}); trying the default backend...")
        sbert = SentenceTransformer(SBERT_MODEL_NAME)
        SEMANTIC_BACKEND = "torch"
    SEMANTIC_AVAILABLE = True
    print(f"Semantic layer ready (backend: {SEMANTIC_BACKEND}).")
except Exception as exc:  # broad on purpose -- import/load failures vary by platform
    SEMANTIC_UNAVAILABLE_REASON = str(exc)
    print(f"WARNING: Semantic layer unavailable ({exc}). Continuing with "
          f"Lexical + Behavioral + Fusion only.")

behavioral_models = {p: joblib.load(os.path.join(MODELS_DIR, f"behavioral_{p}_xgboost.joblib"))
                     for p in T.PLATFORMS}
fusion_models = {p: joblib.load(os.path.join(MODELS_DIR, f"fusion_{p}_logreg.joblib"))
                 for p in T.PLATFORMS}
with open(os.path.join(MODELS_DIR, "platform_behavioral_columns.json")) as f:
    behavioral_columns = json.load(f)
with open(os.path.join(MODELS_DIR, "behavioral_reference_stats.json")) as f:
    reference_stats = json.load(f)  # see compute_reference_stats.py

# Per-word (genuine_freq, fake_freq) in the training corpus -- see
# compute_lexical_word_stats.py. Used only to filter which words
# explain_lexical() surfaces (never changes lexical_score or any prediction):
# LogisticRegression fits coefficients jointly, so an individual word's
# coefficient sign can end up flipped relative to its own raw frequency
# skew; showing that in a "why" explanation reads as flatly wrong even
# though it's an accurate reflection of the joint fit. Optional file --
# missing it just disables the filter (every word passes), same
# graceful-degradation pattern used elsewhere.
LEXICAL_WORD_FREQ_PATH = os.path.join(MODELS_DIR, "lexical_word_class_freq.json")
lexical_word_freq = {}
if os.path.exists(LEXICAL_WORD_FREQ_PATH):
    with open(LEXICAL_WORD_FREQ_PATH) as f:
        lexical_word_freq = json.load(f)
    print(f"Lexical word-frequency stats loaded ({len(lexical_word_freq)} words).")
else:
    print("Lexical word-frequency stats not found -- run compute_lexical_word_stats.py "
          "to filter out frequency-inconsistent words from the Lexical explanation.")

print("Models loaded. Ready.")

BOOL_FIELDS = {"is_verified", "has_website", "has_location",
               "email_verified", "phone_verified", "two_factor_enabled"}
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
    "story_post_rate_weekly": "Stories / week", "retweet_ratio": "Retweet ratio",
    "group_membership_count": "Groups joined", "recommendation_count": "Recommendations",
    "connection_acceptance_rate": "Connection acceptance rate",
    "email_verified": "Email verified", "phone_verified": "Phone verified",
    "two_factor_enabled": "Two-factor enabled", "device_count_30d": "Devices (30d)",
    "login_ip_diversity_30d": "Login IP diversity (30d)",
    "signup_to_first_post_hours": "Signup to first post (hrs)",
    "posting_time_entropy": "Posting time entropy", "follower_growth_rate_7d": "Follower growth (7d)",
    "reports_received_count": "Reports received", "content_removed_count": "Content removed",
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
# Every demo profile (all ~200/platform, shuffled) -- the frontend's compact
# "try examples" row shows just the first few (see app.js renderExampleChips,
# .slice(0, 6)), while the "Browse demo usernames" expandable panel shows this
# whole list per platform (renderExpandedChips, unsliced), so it's a genuine
# browse of the full ~800-profile demo set, not a small fixed sample.
for p in T.PLATFORMS:
    prof = demo_profiles[p].sample(frac=1, random_state=_rng.randint(0, 10_000))  # shuffle
    for _, pick in prof.iterrows():
        SAMPLE_USERNAMES.append({"username": pick["username"], "platform": p,
                                  "is_fake": bool(pick["is_fake"]),
                                  "archetype": str(pick["archetype"])})
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
    back to a cross-platform username search either way.

    Only treats the input as a URL when it already has a scheme, or its first
    path segment is an *exact* match for a known platform domain -- a bare
    username containing a dot (a common, legitimate Instagram-style handle,
    e.g. "first.last") must never be misread as a bare domain like
    "instagram.com" and swallowed into an empty path/empty username."""
    raw = raw.strip()
    first_segment = raw.split("/")[0].lower()
    if "://" not in raw and first_segment not in DOMAIN_TO_PLATFORM:
        return None, raw
    if "://" not in raw:
        raw = "https://" + raw  # allow "instagram.com/foo" without a scheme
    parsed = urlparse(raw)
    platform = DOMAIN_TO_PLATFORM.get(parsed.netloc.lower())
    path_parts = [seg for seg in parsed.path.split("/") if seg]
    if platform == "linkedin" and path_parts and path_parts[0] in ("in", "company"):
        path_parts = path_parts[1:]
    username = path_parts[0] if path_parts else ""
    return platform, username


# ---------------------------------------------------------------------------
# Per-layer, data-driven explanations -- grounded in the model's own math
# (TF-IDF x LogReg coefficients, XGBoost's per-prediction SHAP-style feature
# contributions, the Trust Fusion layer's learned weights). No LLM call: an
# explanation here is never anything the model didn't actually compute.
# ---------------------------------------------------------------------------

def _freq_agrees(word: str, direction: str) -> bool:
    """True if word's raw class-frequency skew (see compute_lexical_word_stats.py)
    agrees with `direction` ('fake' or 'genuine'). A word with no stats (e.g.
    the file is missing, or a rare bigram edge case) is allowed through --
    the filter only ever removes a word it has positive evidence against."""
    stats = lexical_word_freq.get(word)
    if not stats:
        return True
    genuine_freq, fake_freq = stats
    return fake_freq > genuine_freq if direction == "fake" else genuine_freq > fake_freq


def explain_lexical(pooled_text: str, X_lex, lexical_score: float) -> dict:
    feature_names = tfidf.get_feature_names_out()
    coefs = lexical_model.coef_[0]
    coo = X_lex.tocoo()
    contributions = sorted(
        ((feature_names[j], float(v) * float(coefs[j])) for j, v in zip(coo.col, coo.data)),
        key=lambda x: x[1],
    )
    # Only surface a word if its displayed direction also matches its actual
    # frequency pattern in the training data -- see _freq_agrees's docstring
    # and compute_lexical_word_stats.py for why the two can otherwise disagree.
    fake_words = [w for w, c in reversed(contributions) if c > 0 and _freq_agrees(w, "fake")][:5]
    genuine_words = [w for w, c in contributions if c < 0 and _freq_agrees(w, "genuine")][:5]

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
    if not SEMANTIC_AVAILABLE:
        summary = ("Semantic layer unavailable on this machine — PyTorch was blocked at import by a "
                   "system security policy (Windows Application Control/WDAC), not by anything in this "
                   "app. Fusion below uses a neutral 50% placeholder for this layer instead of a real "
                   "score; Lexical and Behavioral are unaffected.")
        return {"summary": summary}
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
    t_start = time.perf_counter()
    captions = list(captions)
    hashtags = list(hashtags)
    pooled_text = " ".join([text] + captions[:T.N_CAPTIONS_FOR_TEXT] + hashtags).strip()
    if not pooled_text:
        pooled_text = "no bio provided"

    X_lex = tfidf.transform([pooled_text])
    lexical_score = float(lexical_model.predict_proba(X_lex)[0, 1])

    if SEMANTIC_AVAILABLE:
        emb = sbert.encode([pooled_text])
        semantic_score = float(semantic_model.predict_proba(emb)[0, 1])
    else:
        semantic_score = 0.5  # neutral placeholder -- see explain_semantic()

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

    analysis_time_seconds = round(time.perf_counter() - t_start, 2)

    return {
        "platform": platform,
        "lexical_score": round(lexical_score, 4),
        "semantic_score": round(semantic_score, 4),
        "semantic_available": SEMANTIC_AVAILABLE,
        "behavioral_score": round(behavioral_score, 4),
        "fusion_score": round(fusion_score, 4),
        "analysis_time_seconds": analysis_time_seconds,
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


@app.route("/")
def index():
    return render_template("index.html", platforms=T.PLATFORMS,
                           text_labels=PLATFORM_TEXT_LABEL, sample_usernames=SAMPLE_USERNAMES,
                           platform_icons=PLATFORM_ICONS, semantic_available=SEMANTIC_AVAILABLE,
                           db_available=db.DB_AVAILABLE)


@app.route("/api/lookup", methods=["GET"])
@limiter.limit("30 per minute")
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


def _db_unavailable():
    return jsonify({"error": f"Database unavailable: {db.DB_UNAVAILABLE_REASON}"}), 503


def _client_id_from(source) -> str | None:
    cid = (source.get("client_id") or "").strip()
    return cid or None


@app.route("/api/history", methods=["GET"])
def get_history():
    if not db.DB_AVAILABLE:
        return _db_unavailable()
    client_id = _client_id_from(request.args)
    if not client_id:
        return jsonify({"error": "client_id is required"}), 400
    return jsonify(db.list_history(client_id))


@app.route("/api/history", methods=["POST"])
@limiter.limit("60 per minute")
def post_history():
    if not db.DB_AVAILABLE:
        return _db_unavailable()
    body = request.get_json(force=True) or {}
    client_id = _client_id_from(body)
    result = body.get("result")
    if not client_id or not result:
        return jsonify({"error": "client_id and result are required"}), 400
    entry = {
        "platform": result.get("platform"), "username": result.get("username"),
        "archetype": result.get("ground_truth_archetype"),
        "ground_truth_is_fake": result.get("ground_truth_is_fake"),
        "fusion_score": result.get("fusion_score"), "result": result,
    }
    return jsonify(db.add_history(client_id, entry))


@app.route("/api/history/<entry_id>", methods=["DELETE"])
@limiter.limit("60 per minute")
def delete_history_one(entry_id):
    if not db.DB_AVAILABLE:
        return _db_unavailable()
    client_id = _client_id_from(request.args)
    if not client_id:
        return jsonify({"error": "client_id is required"}), 400
    db.delete_history_entry(client_id, entry_id)
    return jsonify({"ok": True})


@app.route("/api/history", methods=["DELETE"])
@limiter.limit("20 per minute")
def delete_history_all():
    if not db.DB_AVAILABLE:
        return _db_unavailable()
    client_id = _client_id_from(request.args)
    if not client_id:
        return jsonify({"error": "client_id is required"}), 400
    db.clear_history(client_id)
    return jsonify({"ok": True})


@app.route("/api/saved", methods=["GET"])
def get_saved():
    if not db.DB_AVAILABLE:
        return _db_unavailable()
    client_id = _client_id_from(request.args)
    if not client_id:
        return jsonify({"error": "client_id is required"}), 400
    return jsonify(db.list_saved(client_id))


@app.route("/api/saved", methods=["POST"])
@limiter.limit("60 per minute")
def post_saved():
    if not db.DB_AVAILABLE:
        return _db_unavailable()
    body = request.get_json(force=True) or {}
    client_id = _client_id_from(body)
    result = body.get("result")
    if not client_id or not result:
        return jsonify({"error": "client_id and result are required"}), 400
    entry = {
        "platform": result.get("platform"), "username": result.get("username"),
        "archetype": result.get("ground_truth_archetype"),
        "ground_truth_is_fake": result.get("ground_truth_is_fake"),
        "fusion_score": result.get("fusion_score"), "result": result,
    }
    return jsonify(db.add_saved(client_id, entry))


@app.route("/api/saved/<entry_id>", methods=["DELETE"])
@limiter.limit("60 per minute")
def delete_saved_one(entry_id):
    if not db.DB_AVAILABLE:
        return _db_unavailable()
    client_id = _client_id_from(request.args)
    if not client_id:
        return jsonify({"error": "client_id is required"}), 400
    db.delete_saved_entry(client_id, entry_id)
    return jsonify({"ok": True})


@app.route("/api/saved", methods=["DELETE"])
@limiter.limit("20 per minute")
def delete_saved_all():
    if not db.DB_AVAILABLE:
        return _db_unavailable()
    client_id = _client_id_from(request.args)
    if not client_id:
        return jsonify({"error": "client_id is required"}), 400
    db.clear_saved(client_id)
    return jsonify({"ok": True})


@app.route("/api/scrape", methods=["POST"])
@limiter.limit("5 per minute")
def scrape():
    """
    Scrape a real, public Instagram profile by username or profile URL.

    Ported from the standalone Instagram scrapper project's POST /api/scrape.
    Pure JSON in/out -- no files written to disk, no dataset accumulation.
    """
    data = request.get_json(force=True) or {}
    target = (data.get("username") or data.get("url") or "").strip()
    if not target:
        return jsonify({"error": "a username or profile URL is required"}), 400

    try:
        max_posts = int(data.get("max_posts", 12))
    except (TypeError, ValueError):
        max_posts = 12

    try:
        profile = scrape_instagram_profile(target, max_posts=max_posts)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

    return jsonify({"success": True, "platform": "instagram", "profile": profile})


@app.route("/healthz", methods=["GET"])
@limiter.exempt
def healthz():
    """Liveness/readiness probe -- what's actually loaded and working right
    now, not just "the process is up." No rate limit: orchestrators/uptime
    monitors poll this frequently by design."""
    ok = tfidf is not None and lexical_model is not None and len(behavioral_models) == len(T.PLATFORMS)
    status = {
        "status": "ok" if ok else "degraded",
        "models_loaded": ok,
        "semantic_layer_available": SEMANTIC_AVAILABLE,
        "database_available": db.DB_AVAILABLE,
        "demo_profiles_loaded": sum(len(v) for v in demo_profiles.values()),
    }
    return jsonify(status), 200 if ok else 503


if __name__ == "__main__":
    # debug/reloader off on purpose -- the reloader re-imports this module in a
    # child process, which means loading SBERT + every joblib model twice.
    #
    # Serves via waitress (a production-grade, cross-platform WSGI server)
    # rather than Flask's own dev server, which prints its own warning about
    # not being meant for production if used directly. Same command either
    # way: `python app.py`. Override bind address/port via env vars for a
    # container/cloud deployment (e.g. HOST=0.0.0.0 PORT=8080).
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    from waitress import serve
    print(f"Serving IDGuardian on http://{host}:{port} (waitress, {os.cpu_count() or 4} threads)")
    serve(app, host=host, port=port, threads=os.cpu_count() or 4)
