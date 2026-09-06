# Generation Notes / Assumptions

Per the task instructions, the spec's "Open Items to Confirm" (5,000/50,000 volume,
60/40 genuine/fake balance, and the extra Facebook/LinkedIn fake archetypes) were treated
as already decided and implemented as written. Everything below is a call I had to make
because the spec didn't pin it down exactly, or a correctness issue I found and fixed while
implementing.

## Decisions made

- **`follow_ratio` / `follower_gap` are not stored as CSV columns.** The spec's own
  Instagram/Twitter sections say these are derived features "not stored as raw columns
  unless you want them precomputed," and the documented profile schema tables for both
  platforms omit them. Storing them would also fail the spec's own schema-match check
  (exact column list). They're one line of pandas for whoever does feature extraction next:
  `followers_count / (following_count + 1)` and `followers_count - following_count`.

- **Engagement-rate denominator per platform**, since the spec doesn't pin this down:
  `followers_count` for Instagram/Twitter, `friends_count + followers_count` for Facebook,
  `connections_count` for LinkedIn. Important: the same combined figure is also used as the
  "audience" that drives each post's likes/comments/shares during generation -- otherwise an
  archetype whose reach lives mostly in the secondary count (e.g. a Facebook Page, where
  `followers_count` dwarfs `friends_count`) would get realistic-looking post engagement but
  an absurdly deflated `engagement_rate` once divided by the full audience. Keeping generation
  and the aggregate denominator consistent was necessary for Rule 3 (aggregate consistency)
  to mean anything.

- **Archetype-conditioned numeric ranges** (audience-size ranges, account age, engagement
  rate, comment/share ratios, verification/website/location probabilities, profile
  completion) were hand-authored per archetype based on realistic platform norms, since the
  spec gives qualitative direction ("fake skews younger," "near-1.0 ratios," "low genuine
  engagement relative to audience size") rather than exact numbers. `common.py`'s
  `sample_lognormal_clipped` centers its lognormal draw on the *geometric mean* of the
  given `(low, high)` range rather than on `low` itself -- anchoring at `low` produced
  count distributions that clustered almost entirely near the floor regardless of how large
  `high` was, which is a real bug I caught and fixed by spot-checking generated LinkedIn
  `endorsements_count` (was averaging ~1.2 across a 0-400 range before the fix).

- **Username styles per archetype** (`casual` / `corporate` / `spammy` / `aesthetic` /
  `impersonator`) are an added lightweight signal, not mandated by the spec, but a
  reasonable extra behavioral cue (e.g. spam accounts get `promo_user48213`-style handles,
  impersonators get `name_real`/`name_official` suffixes).

- **LinkedIn `job_title` for the genuine "Business/Company Page" archetype** is set to a
  placeholder ("Company Page" / "Corporate Account") since a company page doesn't have a
  personal job title, and the spec's schema doesn't list `job_title` as nullable.

- **Facebook `Catfish/Romance-Scam Profile` posts always have empty hashtags** -- not
  required by the spec, but consistent with how that archetype actually behaves on the
  platform (private, low-activity, non-promotional).

## Correctness fix worth flagging

- **RNG/Faker seeding is now process-independent.** The first implementation salted the
  global `SEED` per platform using Python's builtin `hash(platform_name)` -- but CPython
  randomizes string hashes per process by default (`PYTHONHASHSEED`), so that would have
  silently produced a *different* dataset on every run despite the fixed `SEED = 42`,
  defeating the spec's reproducibility requirement. Fixed by salting with `zlib.crc32`
  instead, which is stable across processes and machines. Verified by running the full
  generator twice in separate processes and diffing checksums of all 8 CSVs -- byte-identical.

## QA / tuning pass (post-generation)

A separate, deeper QA pass ([`qa_report.py`](qa_report.py)) checked text duplication rates,
numeric distribution sanity, per-feature label leakage, and overall behavioral-feature
separability -- and found the initial generation was unrealistically clean (near-100%
classifier separability, and up to 100% duplicate captions in some archetypes). Two real
bugs (an inverted overlap-ternary, and per-field-only noise that a nonlinear classifier
could route around by fingerprinting archetypes) were found and fixed in `generate_dataset.py`
and `common.py`, not patched onto the CSVs. Full before/after numbers and what changed are in
[`data/qa_report.md`](data/qa_report.md). Install `requirements-qa.txt` (adds scikit-learn +
xgboost on top of the base generator deps) to rerun it.

## Model training pass (`train_models.py`)

Trained the spec's Section 8 architecture: 2 shared models (Lexical: TF-IDF + Logistic
Regression; Semantic: SBERT + XGBoost) + 8 per-platform models (Behavioral x4, Trust Fusion
x4) = 10 total, with a proper out-of-fold stacking methodology so Trust Fusion never sees a
base layer's in-sample predictions (see `data/model_training_report.md` for the full
methodology and numbers).

**A third over-separability bug turned up here, this time in text.** The first full training
run produced Lexical and Semantic layers at 99.9-100% test accuracy/AUC on every platform,
which made Trust Fusion trivially perfect (exactly 1.0/1.0/1.0 everywhere) -- it was just
inheriting the lexical/semantic scores and adding nothing. Root cause: each archetype's
phrase banks use essentially disjoint vocabulary (a spam bank never says what a genuine
bank says), so a bag-of-words or embedding classifier can key off a handful of highly
distinctive tokens and separate the classes almost perfectly -- the QA pass had only tuned
*numeric* separability into a realistic range and explicitly called out text as untested at
the time. Fixed the same way as the earlier numeric fix: added `text_overlap_frac` to
`build_platform_dataset` (`generate_dataset.py`), an independent shadow-archetype swap
applied to bio/headline + captions + hashtags only, decoupled from the numeric
`overlap_frac` since text classifiers are far more sample-efficient at exploiting whatever
overlap-free vocabulary remains. Tuning took two tries: 0.35 overshot badly (61-65%
accuracy, near the 60% majority-class floor with no signal left worth keeping), 0.15 landed
Lexical/Semantic in the target 83-85% band across every platform. Iterated quickly with a
lexical-only check (skipping the slow SBERT re-encode) before committing to a full retrain.

After the fix, every base layer sits in a realistic, non-trivial range, and Trust Fusion
shows a real, positive lift over the best single layer on every platform (+0.006 to +0.078
ROC-AUC) -- the multi-layer architecture is finally doing what it's meant to do.

## Operational note for the next phase (feature extraction)

`hashtags` is stored as an empty string `""` for rows with no hashtags (Facebook/LinkedIn
mostly, per the spec's own note that this is expected). Plain `pd.read_csv(...)` will read
those empty cells back as `NaN`, not `""`, because pandas treats empty CSV fields as missing
by default. Load with `pd.read_csv(path, keep_default_na=False)` for that column, or
`.fillna("")` after loading, so downstream text vectorizers don't choke on `NaN`.
