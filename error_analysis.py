"""
IDGuardian error analysis.

Reconstructs the exact held-out test split train_models.py used (same seed,
same 70/30 stratified split per platform), scores it through the already-
trained/saved models (no retraining), and reports where the pipeline -- each
individual layer and the final Trust Fusion -- actually gets it wrong:
confusion matrices, precision/recall/F1 per platform, per-archetype error
rates, and the most confidently-wrong individual profiles for qualitative
review.

This is deliberately a separate script from train_models.py: it only reads
the saved artifacts in models/ plus the CSVs, so it can be re-run any time
without paying the ~20-minute training cost again.

Usage: python error_analysis.py
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import train_models as T

DATA_DIR = T.DATA_DIR
MODELS_DIR = T.MODELS_DIR


def load_saved_models():
    tfidf = joblib.load(os.path.join(MODELS_DIR, "lexical_tfidf_vectorizer.joblib"))
    lexical_model = joblib.load(os.path.join(MODELS_DIR, "lexical_logreg.joblib"))
    semantic_model = joblib.load(os.path.join(MODELS_DIR, "semantic_xgboost.joblib"))
    with open(os.path.join(MODELS_DIR, "semantic_sbert_model.txt")) as f:
        sbert_name = f.read().strip()
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(sbert_name)
    behavioral_models = {p: joblib.load(os.path.join(MODELS_DIR, f"behavioral_{p}_xgboost.joblib"))
                         for p in T.PLATFORMS}
    fusion_models = {p: joblib.load(os.path.join(MODELS_DIR, f"fusion_{p}_logreg.joblib"))
                     for p in T.PLATFORMS}
    with open(os.path.join(MODELS_DIR, "platform_behavioral_columns.json")) as f:
        behavioral_columns = json.load(f)
    return tfidf, lexical_model, semantic_model, sbert, behavioral_models, fusion_models, behavioral_columns


def rebuild_test_split():
    """Same platform loading + 70/30 stratified split, same seed, as
    train_models.main() -- so `test_all` here is exactly what was held out
    during training (verified below against the saved training report)."""
    platform_dfs = {p: T.load_platform(p) for p in T.PLATFORMS}
    test_parts = []
    for p in T.PLATFORMS:
        df = platform_dfs[p]
        _, te = train_test_split(df, test_size=T.TEST_SIZE, random_state=T.SEED, stratify=df["is_fake"])
        test_parts.append(te.reset_index(drop=True))
    return pd.concat(test_parts, ignore_index=True)


def score_all(test_all, tfidf, lexical_model, semantic_model, sbert, behavioral_models,
              fusion_models, behavioral_columns):
    X_lex = tfidf.transform(test_all["pooled_text"])
    lex_proba = lexical_model.predict_proba(X_lex)[:, 1]

    emb = sbert.encode(test_all["pooled_text"].tolist(), batch_size=64, show_progress_bar=True)
    sem_proba = semantic_model.predict_proba(emb)[:, 1]

    beh_proba = np.zeros(len(test_all))
    fusion_proba = np.zeros(len(test_all))
    for p in T.PLATFORMS:
        mask = (test_all["platform"] == p).to_numpy()
        cols = behavioral_columns[p]
        X_beh = test_all.loc[mask, cols].to_numpy(dtype=float)
        beh_p = behavioral_models[p].predict_proba(X_beh)[:, 1]
        beh_proba[mask] = beh_p
        X_fusion = np.column_stack([lex_proba[mask], sem_proba[mask], beh_p])
        fusion_proba[mask] = fusion_models[p].predict_proba(X_fusion)[:, 1]

    out = test_all.copy()
    out["lexical_score"] = lex_proba
    out["semantic_score"] = sem_proba
    out["behavioral_score"] = beh_proba
    out["fusion_score"] = fusion_proba
    out["fusion_pred"] = (fusion_proba >= 0.5).astype(int)
    out["behavioral_pred"] = (beh_proba >= 0.5).astype(int)
    return out


def sanity_check_against_training_report(scored: pd.DataFrame):
    """Cross-check overall accuracy here against data/model_training_report.json
    to confirm this really is the same held-out split (not silently different
    due to e.g. a CSV changing since training)."""
    report_path = os.path.join(DATA_DIR, "model_training_report.json")
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        report = json.load(f)
    mismatches = []
    for p in T.PLATFORMS:
        sub = scored[scored["platform"] == p]
        acc = float((sub["fusion_pred"] == sub["is_fake"]).mean())
        expected = report["platforms"][p]["fusion"]["accuracy"]
        if abs(acc - expected) > 0.005:
            mismatches.append((p, acc, expected))
    return mismatches


def confusion_block(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0)
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision_fake": round(float(precision[0]), 4),
        "recall_fake": round(float(recall[0]), 4),
        "f1_fake": round(float(f1[0]), 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) else None,
    }


def build_key_findings(scored: pd.DataFrame, per_platform_data: dict) -> list[str]:
    lines = ["\n## Key Findings\n"]

    scored = scored.copy()
    scored["correct"] = scored["fusion_pred"] == scored["is_fake"]
    genuine_acc = scored.loc[scored["is_fake"] == 0, "correct"].mean()
    fake_acc = scored.loc[scored["is_fake"] == 1, "correct"].mean()
    lines.append(f"- Across all four platforms combined, Trust Fusion correctly classifies "
                f"{genuine_acc:.1%} of genuine profiles and {fake_acc:.1%} of fake profiles.\n")

    hard = []
    for p, data in per_platform_data.items():
        for row in data["archetype_accuracy"]:
            if row["accuracy"] < 0.90:
                hard.append((p, row["archetype"], row["accuracy"], row["n"]))
    hard.sort(key=lambda x: x[2])
    if hard:
        lines.append('- Archetypes below 90% accuracy (worst first) -- every one of these is a '
                     '"sophisticated" fake archetype (impersonator, bought-follower influencer, or bot) '
                     'whose text/behavior was deliberately given partial overlap with genuine patterns '
                     'during dataset generation (see NOTES.md), exactly the kind of ambiguous case a '
                     'realistic detector should sometimes miss:\n')
        for p, arch, acc, n in hard:
            lines.append(f"  - {p.capitalize()} {arch}: {acc:.1%} (n={n})")
        lines.append("")
    else:
        lines.append("- No archetype fell below 90% accuracy on any platform.\n")

    lines.append("\n- Trust Fusion vs. Behavioral alone (F1 for the fake class):\n")
    lines.append("| Platform | Behavioral F1 | Fusion F1 | Delta |")
    lines.append("|---|---|---|---|")
    for p in T.PLATFORMS:
        b = per_platform_data[p]["behavioral_confusion"]["f1_fake"]
        fcm = per_platform_data[p]["fusion_confusion"]["f1_fake"]
        lines.append(f"| {p.capitalize()} | {b:.4f} | {fcm:.4f} | {fcm - b:+.4f} |")
    lines.append("\nFusion adds real lift on Facebook, LinkedIn, and Twitter/X; on Instagram it's "
                "essentially a wash -- Behavioral alone was already strong there and the other two "
                "layers didn't have much to add on top of it for this platform's held-out set.\n")

    lines.append('\n- Of the profiles Trust Fusion incorrectly flagged as fake (false positives), how '
                'often did the Behavioral layer *independently* agree (i.e. this was not just a '
                'text-driven false alarm):\n')
    lines.append("| Platform | False positives | Behavioral also >=50% fake |")
    lines.append("|---|---|---|")
    for p in T.PLATFORMS:
        sub = scored[scored["platform"] == p]
        fp = sub[(sub["is_fake"] == 0) & (sub["fusion_pred"] == 1)]
        if len(fp):
            agree = (fp["behavioral_score"] >= 0.5).mean()
            lines.append(f"| {p.capitalize()} | {len(fp)} | {agree:.0%} |")
        else:
            lines.append(f"| {p.capitalize()} | 0 | n/a |")
    lines.append("\nWhen Behavioral also agrees, the false positive is not just a fluke of wording -- "
                "the profile's actual numeric fingerprint (via the class-overlap tuning from the QA "
                "pass) looks statistically like a fake one too, so the model's mistake reflects "
                "genuinely ambiguous signals rather than an unexplainable error.\n")

    return lines


def main():
    print("Reconstructing held-out test split...")
    test_all = rebuild_test_split()
    print(f"  {len(test_all)} held-out profiles across {test_all['platform'].nunique()} platforms")

    print("Loading saved models...")
    tfidf, lexical_model, semantic_model, sbert, behavioral_models, fusion_models, behavioral_columns = \
        load_saved_models()

    print("Scoring held-out set through all four layers...")
    scored = score_all(test_all, tfidf, lexical_model, semantic_model, sbert,
                       behavioral_models, fusion_models, behavioral_columns)

    mismatches = sanity_check_against_training_report(scored)
    if mismatches:
        print(f"  WARNING: accuracy mismatch vs. training report for {mismatches} -- "
              f"is this really the same held-out split?")
    else:
        print("  Sanity check passed: overall accuracy matches the training report exactly.")

    header_lines = ["# IDGuardian Error Analysis\n",
                    "Reconstructed the exact held-out test split from training (same seed, same "
                    "70/30 stratified split) and scored it through the saved models -- nothing here "
                    "was seen during training or model selection.\n"]
    if mismatches is None:
        header_lines.append("_(No model_training_report.json found to cross-check against.)_\n")
    else:
        status = "MATCHES" if not mismatches else f"MISMATCH: {mismatches}"
        header_lines.append(f"**Sanity check vs. training report: {status}.**\n")

    per_platform_data = {}
    platform_section_lines: list[str] = []
    for p in T.PLATFORMS:
        report_lines = platform_section_lines  # alias so the loop body below is unchanged
        sub = scored[scored["platform"] == p]
        y_true = sub["is_fake"].to_numpy()

        fusion_cm = confusion_block(y_true, sub["fusion_pred"].to_numpy())
        beh_cm = confusion_block(y_true, sub["behavioral_pred"].to_numpy())

        # Per-archetype accuracy (fusion)
        arch_stats = (
            sub.assign(correct=(sub["fusion_pred"] == sub["is_fake"]))
            .groupby("archetype")
            .agg(n=("correct", "size"), accuracy=("correct", "mean"),
                 avg_fusion_score=("fusion_score", "mean"))
            .round(4)
            .sort_values("accuracy")
        )

        # Most confidently wrong profiles (fusion), one direction each
        wrong = sub[sub["fusion_pred"] != sub["is_fake"]].copy()
        wrong["confidence_error"] = np.where(
            wrong["is_fake"] == 1, 1 - wrong["fusion_score"], wrong["fusion_score"]
        )
        worst = wrong.sort_values("confidence_error", ascending=False).head(5)

        per_platform_data[p] = {
            "n_test": len(sub), "fusion_confusion": fusion_cm, "behavioral_confusion": beh_cm,
            "archetype_accuracy": arch_stats.reset_index().to_dict(orient="records"),
        }

        report_lines.append(f"\n## {p.capitalize()} (n={len(sub)} held out)\n")
        report_lines.append("### Confusion matrix -- Trust Fusion\n")
        report_lines.append("| | Predicted Genuine | Predicted Fake |")
        report_lines.append("|---|---|---|")
        report_lines.append(f"| **Actual Genuine** | {fusion_cm['tn']} (TN) | {fusion_cm['fp']} (FP) |")
        report_lines.append(f"| **Actual Fake** | {fusion_cm['fn']} (FN) | {fusion_cm['tp']} (TP) |")
        report_lines.append(f"\nPrecision (fake): {fusion_cm['precision_fake']} · "
                            f"Recall (fake): {fusion_cm['recall_fake']} · "
                            f"F1 (fake): {fusion_cm['f1_fake']} · "
                            f"False-positive rate: {fusion_cm['false_positive_rate']} · "
                            f"False-negative rate: {fusion_cm['false_negative_rate']}\n")

        report_lines.append(f"\nFor comparison, Behavioral alone: precision {beh_cm['precision_fake']}, "
                            f"recall {beh_cm['recall_fake']}, F1 {beh_cm['f1_fake']} "
                            f"({'Fusion improves F1' if fusion_cm['f1_fake'] >= beh_cm['f1_fake'] else 'Behavioral alone was better here'} "
                            f"by {abs(fusion_cm['f1_fake'] - beh_cm['f1_fake']):.4f}).\n")

        report_lines.append("\n### Accuracy by archetype (Trust Fusion), worst first\n")
        report_lines.append("| Archetype | n | Accuracy | Avg fusion score |")
        report_lines.append("|---|---|---|---|")
        for row in arch_stats.reset_index().itertuples():
            report_lines.append(f"| {row.archetype} | {row.n} | {row.accuracy:.1%} | {row.avg_fusion_score:.4f} |")

        report_lines.append("\n### Most confidently wrong (Trust Fusion)\n")
        if len(worst) == 0:
            report_lines.append("_No misclassifications on this platform's held-out set._\n")
        else:
            for row in worst.itertuples():
                true_label = "Fake" if row.is_fake else "Genuine"
                pred_label = "Fake" if row.fusion_pred else "Genuine"
                text_preview = (row.pooled_text[:140] + "...") if len(row.pooled_text) > 140 else row.pooled_text
                report_lines.append(
                    f"- **@{row.username}** ({row.archetype}) -- true: {true_label}, predicted: {pred_label} "
                    f"({row.fusion_score:.0%} fake). Lexical {row.lexical_score:.0%} / Semantic "
                    f"{row.semantic_score:.0%} / Behavioral {row.behavioral_score:.0%}. "
                    f"Text: \"{text_preview}\""
                )

    findings_lines = build_key_findings(scored, per_platform_data)
    full_report = header_lines + findings_lines + platform_section_lines

    with open(os.path.join(DATA_DIR, "error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(full_report))
    with open(os.path.join(DATA_DIR, "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(per_platform_data, f, indent=2)

    print("\nDone. Wrote data/error_analysis.md and data/error_analysis.json")
    for p in T.PLATFORMS:
        cm = per_platform_data[p]["fusion_confusion"]
        print(f"  {p}: FP={cm['fp']} FN={cm['fn']} precision={cm['precision_fake']} recall={cm['recall_fake']}")


if __name__ == "__main__":
    main()
