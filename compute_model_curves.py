"""
Compute ROC curve, precision-recall curve, and calibration (reliability
diagram) data for the Trust Fusion layer, per platform and pooled overall.

Reuses error_analysis.py's exact held-out-test-set reconstruction (same
seed, same 70/30 stratified split) and model scoring -- nothing here is
computed on data the models saw during training. Read-only: loads saved
models, writes one small JSON artifact. Does not touch app.py's live
scoring path or any existing model file.

Output: models/model_curves.json, served by GET /api/model_performance.

Usage: python compute_model_curves.py   (run after train_models.py)
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

import error_analysis as E
import train_models as T

MODELS_DIR = T.MODELS_DIR
N_ROC_POINTS = 51
N_PR_POINTS = 51
N_CALIBRATION_BINS = 10


def curve_block(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = float(auc(fpr, tpr))
    grid_fpr = np.linspace(0, 1, N_ROC_POINTS)
    grid_tpr = np.interp(grid_fpr, fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    # sklearn returns recall descending (thresholds increasing) -- reverse to
    # ascending so np.interp's xp requirement (increasing) is satisfied.
    recall_asc = recall[::-1]
    precision_asc = precision[::-1]
    pr_auc = float(auc(recall, precision))
    grid_recall = np.linspace(0, 1, N_PR_POINTS)
    grid_precision = np.interp(grid_recall, recall_asc, precision_asc)

    # Calibration / reliability diagram: bucket predicted probability into
    # N_CALIBRATION_BINS equal-width bins, compare mean predicted probability
    # in each bin against the actual observed fake-rate in that bin. A
    # well-calibrated model's points sit near the y=x diagonal.
    bin_edges = np.linspace(0, 1, N_CALIBRATION_BINS + 1)
    bin_idx = np.clip(np.digitize(y_proba, bin_edges[1:-1]), 0, N_CALIBRATION_BINS - 1)
    bin_centers, bin_predicted, bin_actual, bin_counts = [], [], [], []
    for b in range(N_CALIBRATION_BINS):
        mask = bin_idx == b
        n = int(mask.sum())
        bin_centers.append(round(float((bin_edges[b] + bin_edges[b + 1]) / 2), 3))
        bin_counts.append(n)
        if n > 0:
            bin_predicted.append(round(float(y_proba[mask].mean()), 4))
            bin_actual.append(round(float(y_true[mask].mean()), 4))
        else:
            bin_predicted.append(None)
            bin_actual.append(None)

    return {
        "n": int(len(y_true)), "n_fake": int(y_true.sum()), "n_genuine": int((1 - y_true).sum()),
        "roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4),
        "roc": {"fpr": [round(float(v), 4) for v in grid_fpr],
                "tpr": [round(float(v), 4) for v in grid_tpr]},
        "pr": {"recall": [round(float(v), 4) for v in grid_recall],
               "precision": [round(float(v), 4) for v in grid_precision]},
        "calibration": {"bin_center": bin_centers, "predicted": bin_predicted,
                         "actual": bin_actual, "count": bin_counts},
    }


def main():
    print("Reconstructing held-out test split...")
    test_all = E.rebuild_test_split()

    print("Loading saved models...")
    tfidf, lexical_model, semantic_model, sbert, behavioral_models, fusion_models, behavioral_columns = \
        E.load_saved_models()

    print("Scoring held-out set through all four layers...")
    scored = E.score_all(test_all, tfidf, lexical_model, semantic_model, sbert,
                          behavioral_models, fusion_models, behavioral_columns)

    out = {"platforms": {}, "overall": {}}
    for p in T.PLATFORMS:
        sub = scored[scored["platform"] == p]
        out["platforms"][p] = curve_block(sub["is_fake"].to_numpy(), sub["fusion_score"].to_numpy())
        print(f"  {p}: ROC-AUC={out['platforms'][p]['roc_auc']} PR-AUC={out['platforms'][p]['pr_auc']}")

    out["overall"] = curve_block(scored["is_fake"].to_numpy(), scored["fusion_score"].to_numpy())
    print(f"  overall: ROC-AUC={out['overall']['roc_auc']} PR-AUC={out['overall']['pr_auc']}")

    out_path = os.path.join(MODELS_DIR, "model_curves.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
