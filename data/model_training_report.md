# IDGuardian Model Training Report

10 models trained per DATASET_GENERATION_SPEC.md Section 8: 2 shared (Lexical, Semantic) + 8 per-platform (Behavioral x4, Fusion x4).

## Methodology

- Per-platform 70/30 train/test split, stratified on `is_fake`, seed=42.
- Lexical (TF-IDF + LogisticRegression) and Semantic (SBERT + XGBoost) are fit once on all four platforms' pooled training text.
- Behavioral (XGBoost) and Trust Fusion (LogisticRegression) are fit separately per platform.
- Trust Fusion is trained on 5-fold out-of-fold predictions from the other three layers (never on their in-sample predictions), so its reported lift over any single layer isn't inflated by a layer seeing its own answers.
- All metrics below are computed on the held-out test set only.

## Shared layers (pooled across all 4 platforms)

| Layer | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Lexical | 0.8495 | 0.8186 | 0.8469 |
| Semantic | 0.8473 | 0.8152 | 0.8511 |

## Per-platform results

### Instagram (train=14000, test=6000)

| Layer | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Lexical | 0.8477 | 0.8167 | 0.8415 |
| Semantic | 0.8465 | 0.8148 | 0.845 |
| Behavioral | 0.8947 | 0.8644 | 0.9574 |
| Fusion | 0.9363 | 0.92 | 0.9781 |

Fusion vs. best single layer (ROC-AUC): 0.9781 vs. 0.9574 (+0.0207)

### Facebook (train=14000, test=6000)

| Layer | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Lexical | 0.8518 | 0.8215 | 0.8528 |
| Semantic | 0.8498 | 0.8185 | 0.8585 |
| Behavioral | 0.9297 | 0.9122 | 0.9744 |
| Fusion | 0.9442 | 0.9298 | 0.9866 |

Fusion vs. best single layer (ROC-AUC): 0.9866 vs. 0.9744 (+0.0122)

### Linkedin (train=14000, test=6000)

| Layer | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Lexical | 0.8498 | 0.8193 | 0.8447 |
| Semantic | 0.8453 | 0.8123 | 0.8511 |
| Behavioral | 0.9668 | 0.9587 | 0.9949 |
| Fusion | 0.974 | 0.9676 | 0.9961 |

Fusion vs. best single layer (ROC-AUC): 0.9961 vs. 0.9949 (+0.0012)

### Twitter (train=14000, test=6000)

| Layer | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Lexical | 0.8485 | 0.8168 | 0.8485 |
| Semantic | 0.8475 | 0.8152 | 0.8494 |
| Behavioral | 0.9232 | 0.9011 | 0.9736 |
| Fusion | 0.9477 | 0.9336 | 0.9867 |

Fusion vs. best single layer (ROC-AUC): 0.9867 vs. 0.9736 (+0.0131)
