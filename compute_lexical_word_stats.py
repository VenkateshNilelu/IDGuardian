"""
Precompute, per Lexical-vocabulary word, its raw document frequency among
genuine vs. fake training profiles -- used to filter the Lexical explanation
UI so it never shows a word whose displayed direction (fake-leaning /
genuine-leaning, from the trained LogisticRegression coefficient) contradicts
its actual frequency pattern in the training data.

Why this is needed: LogisticRegression fits all vocabulary coefficients
jointly, not word-by-word, so an individual word's coefficient sign can end
up flipped relative to its own raw frequency skew (e.g. "freeprize" lives
exclusively in the Spam Account archetype's hashtag pool, yet the trained
model gave it a *negative* -- genuine-leaning -- coefficient). That's not a
bug in the model or the app: it's an accurate reflection of the joint fit.
But surfacing it in a "why did this score the way it did" explanation reads
as flatly wrong, so app.py's explain_lexical() cross-checks against this
file before showing a word, and drops it if the two disagree.

This only affects what's *displayed* -- it does not change lexical_score,
the trained coefficients, or any other layer's behavior.

Output: models/lexical_word_class_freq.json
Usage: python compute_lexical_word_stats.py   (run after train_models.py)
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

import train_models as T

MODELS_DIR = T.MODELS_DIR


def main():
    print("Loading platform data...")
    all_df = pd.concat([T.load_platform(p) for p in T.PLATFORMS], ignore_index=True)
    is_fake = all_df["is_fake"].to_numpy()
    n_genuine = int((is_fake == 0).sum())
    n_fake = int((is_fake == 1).sum())
    print(f"  {len(all_df)} profiles ({n_genuine} genuine, {n_fake} fake)")

    tfidf = joblib.load(os.path.join(MODELS_DIR, "lexical_tfidf_vectorizer.joblib"))
    print("Transforming pooled text with the trained TF-IDF vectorizer...")
    X = tfidf.transform(all_df["pooled_text"])

    presence = (X > 0)  # sparse boolean: did this doc contain this vocab word at all
    genuine_doc_count = np.asarray(presence[is_fake == 0].sum(axis=0)).ravel()
    fake_doc_count = np.asarray(presence[is_fake == 1].sum(axis=0)).ravel()
    genuine_freq = genuine_doc_count / max(n_genuine, 1)
    fake_freq = fake_doc_count / max(n_fake, 1)

    feature_names = tfidf.get_feature_names_out()
    stats = {
        feature_names[i]: [round(float(genuine_freq[i]), 5), round(float(fake_freq[i]), 5)]
        for i in range(len(feature_names))
    }

    out_path = os.path.join(MODELS_DIR, "lexical_word_class_freq.json")
    with open(out_path, "w") as f:
        json.dump(stats, f)
    print(f"Wrote {out_path} ({len(stats)} vocabulary words)")


if __name__ == "__main__":
    main()
