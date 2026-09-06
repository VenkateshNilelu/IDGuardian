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
| Lexical | 0.847 | 0.8166 | 0.8472 |
| Semantic | 0.843 | 0.8105 | 0.8497 |

## Per-platform results

### Instagram (train=3500, test=1500)

| Layer | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Lexical | 0.8493 | 0.8189 | 0.8496 |
| Semantic | 0.8473 | 0.8161 | 0.8453 |
| Behavioral | 0.8673 | 0.8346 | 0.8872 |
| Fusion | 0.876 | 0.8324 | 0.9543 |

Fusion vs. best single layer (ROC-AUC): 0.9543 vs. 0.8872 (+0.0671)

### Facebook (train=3500, test=1500)

| Layer | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Lexical | 0.848 | 0.8185 | 0.8482 |
| Semantic | 0.8453 | 0.8135 | 0.8555 |
| Behavioral | 0.8753 | 0.8448 | 0.9501 |
| Fusion | 0.928 | 0.9091 | 0.9753 |

Fusion vs. best single layer (ROC-AUC): 0.9753 vs. 0.9501 (+0.0252)

### Linkedin (train=3500, test=1500)

| Layer | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Lexical | 0.8387 | 0.8064 | 0.8392 |
| Semantic | 0.8327 | 0.7977 | 0.8479 |
| Behavioral | 0.934 | 0.9181 | 0.981 |
| Fusion | 0.9547 | 0.9434 | 0.9869 |

Fusion vs. best single layer (ROC-AUC): 0.9869 vs. 0.9810 (+0.0059)

### Twitter (train=3500, test=1500)

| Layer | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Lexical | 0.852 | 0.8227 | 0.8514 |
| Semantic | 0.8467 | 0.8148 | 0.8501 |
| Behavioral | 0.8593 | 0.8272 | 0.8752 |
| Fusion | 0.886 | 0.8496 | 0.9532 |

Fusion vs. best single layer (ROC-AUC): 0.9532 vs. 0.8752 (+0.0780)
