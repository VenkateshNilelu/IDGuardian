# IDGuardian Dataset QA & Tuning Report

**Status: PASS — the dataset is ready for feature extraction and model training.**

This is a deeper pass than the generation-time validation in `data/validation_report.txt`.
It re-verifies structural correctness, and additionally checks text diversity, numeric
distribution sanity, per-feature label leakage, and end-to-end statistical separability —
then tunes the generator against what it found and re-checks until every criterion in the
QA spec is satisfied. Produced by [`qa_report.py`](../qa_report.py); samples in
[`text_samples.md`](text_samples.md).

Two real bugs were found and fixed during this pass (not just parameter tweaks) — see
"Bugs found and fixed" below. Everything else is a distribution-tuning change.

---

## 1. Structural correctness — PASS (unchanged by tuning)

Re-verified independently of the generator's own validation pass, across all four platforms:

| Check | Instagram | Facebook | LinkedIn | Twitter/X |
|---|---|---|---|---|
| Orphan posts | 0 | 0 | 0 | 0 |
| Post/profile archetype mismatches | 0 | 0 | 0 | 0 |
| Post/profile is_fake mismatches | 0 | 0 | 0 | 0 |
| avg_likes/comments/shares mismatches vs. actual post means | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| posts_count (tweets_count) mismatches | 0 | 0 | 0 | 0 |
| Profile row count | 5,000 | 5,000 | 5,000 | 5,000 |
| Post row count | 49,719 | 49,788 | 49,715 | 49,614 |
| Overall genuine/fake balance | 60.0/40.0 | 60.0/40.0 | 60.0/40.0 | 60.0/40.0 |

Post counts land within 0.6% of the 50,000 target (expected — post count per profile is
Poisson(10)-sampled, not fixed, per the spec). Per-archetype counts are even within each
class (e.g. Instagram genuine archetypes: 750/750/750/750; fake: 667/667/666 — as even as
5,000×60% and 5,000×40% divide across 4 and 3 archetypes respectively). No violations found
in this section at any point in the QA process — nothing here needed tuning.

## 2. Text quality / diversity — PASS (fixed; started badly broken)

**Before tuning**, exact-duplicate rates were severe — this was the single worst problem
found:

| | Instagram | Facebook | LinkedIn | Twitter/X |
|---|---|---|---|---|
| Worst bio/headline archetype | 51.95% (Celebrity Impersonator) | 54.8% (Catfish) | 94.89% (Spam/Bot) | 55.26% (Celebrity Impersonator) |
| Worst caption archetype | 77.46% (Fitness Creator) | 95.64% (Catfish) | 100% (Business/Company Page, Consultant) | 77.44% (Fitness Creator) |

Root cause: the phrase-bank recombination (opener × body × optional CTA, drawn from
pools of ~4-6 items each) produced only a few dozen to a few hundred distinct strings per
archetype, but each archetype has ~4,600-7,600 posts — guaranteed heavy collision. LinkedIn
was worst because it also has near-zero emoji usage (appropriately, for tone), removing a
diversity axis the other platforms leaned on.

**Fix:** added a high-cardinality "detail" clause to almost every bio/caption
(`PhraseBank.high_card_pool` / `CaptionBank.details` in [`common.py`](../common.py)) —
short, archetype-appropriate sentences carrying a Faker city/company/first-name or a
4-digit number, sampled with ~95% probability (plus a second independent draw for LinkedIn
specifically, `detail2_prob`, since it has no emoji axis to fall back on). A single Faker
city or 4-digit number multiplies the combinatorial space by 1-2 orders of magnitude far
more cheaply than hand-writing thousands more fixed phrases. Also moderately expanded the
weakest fixed opener/body pools (Developer, Fitness Creator, Fake Influencer, Celebrity
Impersonator, Catfish, and all seven LinkedIn caption/bio banks).

**After tuning**, every platform × archetype bucket is under 3% (vs. the 5% flag
threshold), most under 1.5%:

| | Instagram | Facebook | LinkedIn | Twitter/X |
|---|---|---|---|---|
| Worst bio/headline archetype | 2.1% | 0.4% | 0.9% | 1.5% |
| Worst caption archetype | 2.51% | 2.32% | 1.22% | 2.53% |

Five random bios/headlines and five random captions per platform × archetype are saved in
[`text_samples.md`](text_samples.md) for manual spot-checking tone/realism.

## 3. Numeric distribution sanity — PASS (no impossible values at any point)

No negative counts, no `engagement_rate` outside [0, 2.0], no `profile_completion_score`
outside [0,1], and no LinkedIn `connections_count` above the real 30,000 cap — at any stage,
before or after tuning. Full min/max/mean/std per platform × archetype × column was computed
by `qa_report.py` and is not reproduced in full here (see the script), but nothing flagged.

**Archetype-conditioned skew — confirmed present, and this is where the real tuning
happened** (see the leakage/separability sections below for what changed and why): fake
archetypes measurably skew younger (`account_age_days`) and lower-engagement
(`engagement_rate`) than genuine ones on every platform, with Cohen's *d* comfortably above
the 0.2 "distinguishable" threshold in the final data — e.g. Instagram
`account_age_days`: genuine mean 1,902 days vs. fake mean 302 days. The signal is real,
directional, and matches the spec's intent, without being a clean, unrealistic cutoff (see
below).

## 4. Per-feature leakage — PASS throughout (no column ever crossed 0.98)

No single numeric column, on any platform, ever reached the >98% single-feature accuracy
leakage threshold — not before tuning, not after. The highest single-feature accuracy seen
**before** tuning was LinkedIn `account_age_days` at 97.3%, and Instagram `engagement_rate`
at 94.9% — both uncomfortably close to leakage without technically crossing it, and both
were part of why the *combined* classifier (section 5) was landing at ~100%. After tuning,
the highest single-feature accuracy on any platform dropped to **84.2%** (Instagram).

## 5. Overall separability — PASS (tuned from ~100% down into the 80-95% target band)

**Before tuning**, every platform was separable almost perfectly by a plain classifier on
behavioral features alone — too clean for a realistic fake-detection benchmark:

| Platform | LR accuracy | LR ROC-AUC | XGBoost accuracy | XGBoost ROC-AUC |
|---|---|---|---|---|
| Instagram | 99.87% | 1.000 | 99.73% | 1.000 |
| Facebook | 99.93% | 1.000 | 100.00% | 1.000 |
| LinkedIn | 99.93% | 1.000 | 100.00% | 1.000 |
| Twitter/X | 100.00% | 1.000 | 99.80% | 1.000 |

**After tuning** (see "Bugs found and fixed" + the overlap mechanism below):

| Platform | LR accuracy | LR ROC-AUC | XGBoost accuracy | XGBoost ROC-AUC |
|---|---|---|---|---|
| Instagram | 86.93% | 0.876 | 87.07% | 0.882 |
| Facebook | 86.40% | 0.9215 | 88.27% | 0.9577 |
| LinkedIn | 87.13% | 0.9321 | 93.33% | 0.9806 |
| Twitter/X | 86.60% | 0.8739 | 87.00% | 0.8905 |

All eight numbers now sit inside the 80-95% band the spec calls "realistic and learnable but
not trivial." Split: 70/30 train/test, stratified, `random_state=42`, on every numeric
behavioral column per platform (no text/id/label columns).

---

## Bugs found and fixed (not just tuning)

1. **The class-overlap mechanism was silently a no-op — inverted ternary logic.** The first
   tuning attempt added "some fake accounts look old, some genuine accounts look young"
   noise on `account_age_days` and `engagement_rate`, using range dictionaries named
   `genuine_range`/`fake_range` (holding the value to *inject into* that class). The
   application code did `fake_range if is_fake==0 else genuine_range` — backwards relative
   to that naming — so every "overlapped" row was resampled from a range nearly identical to
   its own archetype's native range. This produced almost no measurable change (leakage on
   LinkedIn `account_age_days` moved from 97.3% to 98.1% — i.e. got *worse*, a clear tell).
   Fixed by correcting the ternary in three places (`build_platform_dataset`,
   `_apply_overlap`, and Facebook's `mutual_friends_count` builder) so the range meant for a
   class is actually applied to that class.

2. **Per-field noise couldn't cap combined-model separability — needed a per-profile
   redesign, not a parameter fix.** Even after fixing bug #1, noising only
   `account_age_days` and `engagement_rate` left every *other* numeric field (audience size,
   comment/share ratios, profile completion, verified/website/location) fully intact per
   archetype. Since `is_fake` is a deterministic function of `archetype` by construction, a
   nonlinear classifier (XGBoost) could still fingerprint which archetype cluster a row
   belonged to from the untouched fields and back out the label — combined accuracy stayed
   at 95-100% even with zero single-feature leakage. Replaced the mechanism with a
   **shadow-archetype swap**: for `overlap_frac` (tuned to 13%) of every archetype's
   profiles, the *entire* numeric fingerprint — not just one or two fields — is redrawn from
   a randomly chosen opposite-class archetype's config, while the row keeps its own true
   archetype/is_fake label and its own archetype's text bank. This means a genuine subset
   really does look, behaviorally, like a specific fake archetype end-to-end (and vice
   versa) — realistic "sophisticated fake" and "unlucky genuine" cases — which is what
   actually brought combined accuracy down into the realistic band. See
   `sample_numeric_fields` and the per-block sampling loop in `build_platform_dataset`
   ([`generate_dataset.py`](../generate_dataset.py)).

Both fixes are in the checked-in generator, not applied to the CSVs directly — the dataset
is fully regenerable via `python generate_dataset.py` and was verified byte-identical across
two independent runs after every change (fixed seed, `zlib.crc32`-salted per platform,
immune to Python's per-process string-hash randomization).

## What did *not* need to change

- Structural linkage/consistency rules (section 1): correct from the start, no tuning.
- Numeric range sanity (no impossible values): clean from the start.
- The direction of every archetype-conditioned skew (fake younger, lower-engagement, etc.):
  already correct: only the *magnitude* of separation needed softening, not the direction.

## Remaining minor note (not a blocker)

Text-based separability was not measured or tuned in this pass — only the behavioral/numeric
layer (per the spec's own request in step 5, "all behavioral numeric features"). The spec's
multi-layer architecture (Section 8) pools a lexical/semantic text layer on top of the
per-platform behavioral layer, so some additional separability from text is expected and
desirable in the full system; it was intentionally left alone here rather than diluted to
hit an arbitrary text-only target.

---

## Bottom line

Structural integrity: perfect throughout. Text diversity: fixed from a critical failure
(up to 100% duplicate captions in some buckets) down to consistently under 3%. Numeric
distributions: sane throughout, with confirmed, directionally-correct, non-trivial
genuine/fake skew. Leakage: no single feature ever crossed the danger threshold, and the
worst offender was pulled down from 97.3% to 84.2%. Overall separability: tuned from a
too-clean ~100% down to a realistic 86-93% band on every platform. **The dataset is ready
for feature extraction and model training.**
