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

## Scoring web app (`app.py`)

A small Flask app (`app.py` + `templates/index.html` + `static/`) loads all 10 trained
models once at startup and exposes a single-page UI + `POST /api/predict` JSON endpoint
to score one profile at a time through the full pipeline (platform tabs switch which
behavioral fields and text label are shown). Run with `python app.py` after installing
`requirements-webapp.txt`, then open `http://127.0.0.1:5000`. It's a local Flask dev
server (explicitly not for production use, and not deployed anywhere) — built as the
demo/manual-testing surface for the trained models, not a scraping or moderation tool.
Verified end-to-end with both a synthetic spam-styled profile (scored 55% fake) and a
synthetic catfish-styled profile (scored 96% fake) against a clean genuine-student profile
(scored 3% fake).

## Demo lookup dataset + one-box UI (`generate_demo_dataset.py`)

Added a username-lookup feature so the web app's primary flow is a single input
box ("enter a username") instead of manually filling in a profile's fields --
without ever touching a real platform. A tool (agent-reach) was proposed for
actually scraping Instagram/Facebook/LinkedIn to power this; declined, since it
works via browser automation on a real logged-in account (its own README warns
of "risk of platform detection and account suspension") and violates all three
platforms' Terms of Service -- exactly the kind of real-user-data collection
this project has avoided from the start. Built the same UX with synthetic data
instead.

`generate_demo_dataset.py` generates 200 profiles/platform (800 total) using
the same generator and archetype-conditioned realism as the main dataset, but
with a distinct RNG/Faker salt (`"<platform>_demo"`) and user_id prefix
(`igdemo_` etc.) so these profiles are guaranteed disjoint from the
5,000/platform set the models were trained and tested on -- a lookup is
therefore an honest generalization check, not a replay of a memorized
example. Verified zero `user_id` overlap with the main dataset (a handful of
coincidental Faker *username* collisions exist, which is harmless since
lookup only ever searches the demo set).

`app.py` loads this demo set at startup, indexes it by lowercased username,
and exposes `GET /api/lookup?username=...`: on a match it pulls that
profile's bio/captions/hashtags and behavioral fields, runs the same
`score_profile()` the manual form uses, and additionally returns the row's
*true* archetype/is_fake so the UI can show "Ground truth: X" next to the
prediction -- a transparency touch that doubles as a live accuracy check.
The manual-entry form still exists (demoted to a collapsed "Advanced" section)
and still calls the original `/api/predict`.

Found and fixed a real CSS bug while testing this: `.results { display: flex }`
silently overrode the browser's default `[hidden] { display: none }` rule
(author styles always beat user-agent defaults), so the results and error
panels were visible on first page load before any submission. Fixed with a
global `[hidden] { display: none !important; }` rule.

## Per-layer explanations (grounded, not LLM-generated)

The user asked for the UI to explain *why* each layer scored a profile the way
it did, and mentioned using "GPT" for this. Clarified first: this app has no
usable LLM API key available to it (Claude Code's own auth doesn't hand
backend scripts a callable key), and offered a choice between wiring one in
(cost/latency/key management) versus data-driven template explanations
computed directly from each model's own math. User chose the latter.

Implementation, per layer:
- **Lexical**: for the profile's actual TF-IDF vector, multiply each non-zero
  term by the logistic regression's coefficient for that term -- the top
  positive/negative contributions are literally which words swung the score.
- **Semantic**: SBERT's 384 embedding dimensions aren't individually
  interpretable, so this stays qualitative (score + agreement/disagreement
  with the lexical layer) rather than claiming a false level of precision.
- **Behavioral**: XGBoost's booster supports true per-prediction SHAP-style
  contributions natively (`booster.predict(dmatrix, pred_contribs=True)`) --
  no extra library needed. Ranked by |contribution| and paired with
  "genuine-typical" / "fake-typical" reference values precomputed from the
  training set (`compute_reference_stats.py` -> `models/behavioral_reference_stats.json`,
  a small checked-in artifact so the app never needs the full, gitignored
  training CSVs at runtime).
- **Fusion**: its own logistic regression only has 3 inputs, so its learned
  `coef_` directly says which layer it trusts most for that platform.

None of this calls out to Claude/GPT/any external model -- every explanation
is a direct readout of numbers the pipeline already computed, so there's
nothing to hallucinate a reason that isn't what actually happened.

## Single-box lookup: username or URL, structured profile display

The lookup box now accepts a bare demo username *or* a full profile URL
(instagram.com/facebook.com/linkedin.com (`/in/`, `/company/`)/twitter.com
and x.com) -- `parse_profile_input()` in app.py extracts platform + username
from the URL when present, otherwise searches the demo set by username alone
across all four platforms. After a lookup, the UI shows exactly what was
fetched (bio/headline, every caption used, hashtags, and every behavioral
field with its label) before showing the score -- so "what did the model
actually see" is never a black box.

## Dashboard redesign: sidebar shell, glassmorphism, brand icons

The user shared a reference screenshot of a dashboard-style layout (sidebar
nav, top bar, hero search with platform branding, "try examples" chips,
quick-action cards, bottom platform switcher) and asked for that instead of
the earlier single-column card layout. Rebuilt `templates/index.html` +
`static/style.css` around that shell:

- Sidebar (Home / Analysis History / Reports / Settings / Help & Support) --
  only Home does anything real; the others show an honest inline toast
  ("isn't wired up in this research demo") rather than pretending to be full
  pages, since building real history/reports/settings screens wasn't asked
  for and would be misleading to fake.
- "Upgrade to Pro" from the reference became a plain "Research demo /
  synthetic data only, no paid tier" card -- kept the visual slot but didn't
  imply a real subscription tier that doesn't exist.
- Manual entry (previously a `<details>` element) is now a Quick Action card
  that toggles the same form; still posts to the unchanged `/api/predict`.
- Platform switcher pills drive `selectPlatform()` in app.js, which updates
  the hero icon/title/gradient, the input placeholder, the per-platform
  example chips (from `SAMPLE_USERNAMES`, bumped to 2 genuine + 2 fake per
  platform to fill the row out), and the manual form's visible fields --
  all client-side, no extra request.

Then two follow-up refinement requests, both purely visual/CSS + a small
icon-plumbing change, no backend logic touched:

- **"proper glassmorphism, make it look luxurious"**: switched every panel
  (sidebar, cards, search bar, chips, platform switcher) to semi-transparent
  backgrounds with `backdrop-filter: blur(22px) saturate(180%)`, added a
  fixed colorful gradient-mesh background (radial gradients) behind
  everything so the blur has something to blur, added a `--gold` gradient
  variable used sparingly (tagline, active-nav accent bar, plan label) for a
  premium accent, switched the font to Plus Jakarta Sans (Google Fonts), and
  gave cards/buttons soft lift shadows + hover transitions. Both light and
  dark theme variants (toggle in the top bar, persisted to localStorage)
  define their own glass/mesh tokens.
- **"use original icons of platforms"**: replaced the emoji/text placeholders
  (📷, plain "f", "in", ✖) with small inline SVG brand marks in the same
  style commonly shipped in open icon libraries (Simple Icons / Font
  Awesome) -- recognizable simplified glyphs for identifying which platform
  a button relates to, not exact proprietary logo files. Defined once in
  `app.py` (`PLATFORM_ICONS`), rendered server-side for the static platform-
  switcher pills (Jinja `|safe`) and exposed via `window.PLATFORM_ICONS` for
  the JS-driven hero icon swap, so there's one source of truth instead of
  duplicating markup between template and JS.

Hit one real snag while testing this: edits to `app.py` don't take effect
until the Flask process restarts (debug/reloader is off on purpose -- see
the "Scoring web app" note above), so after adding `PLATFORM_ICONS` the
already-running dev server kept serving the old template context
(`window.PLATFORM_ICONS` empty, icons blank) until restarted. Worth
remembering for any future backend change: restart, don't just reload.

## Dataset v2: scale, India localization, platform-owner telemetry

User asked to scale up (5,000 -> 20,000 profiles/platform, ~50k -> ~200k posts/platform),
localize the dataset to Indian users, and add more parameters -- "think like you are the
owner of that platform" -- to make the models more accurate.

- **Scale**: `common.TARGET_PROFILES_PER_PLATFORM`/`TARGET_POSTS_PER_PLATFORM` bumped 4x.
- **Locale**: `common.make_faker()` now runs `Faker(locale="en_IN")`, so every
  `{name}`/`{first}`/`{city}`/`{handle}` fill across all four platforms' phrase banks
  (~1,400 lines) became Indian for free -- did not need to hand-rewrite the phrase banks
  themselves, since they're all filled via the shared `fill_template()` helper. Faker's
  `en_IN` `company()` provider still leans Western-suffixed ("Kale LLC"), so `{company}`
  was switched to a curated `INDIAN_COMPANY_NAMES` pool instead. City/state are sampled
  together from `INDIAN_CITY_STATE_PAIRS` so they can't mismatch. Added `{festival}`/
  `{cricket}`/`{food}` placeholders and used them in a handful of student/business/fitness
  phrase-bank fragments for cultural flavor (not exhaustively -- scope tradeoff, see below).
- **New columns** ("platform owner" internal telemetry -- signals a real platform's trust &
  safety systems would log that no public scraping API ever exposes): `email_verified`,
  `phone_verified`, `two_factor_enabled`, `device_count_30d`, `login_ip_diversity_30d`,
  `signup_to_first_post_hours`, `posting_time_entropy`, `follower_growth_rate_7d`,
  `reports_received_count`, `content_removed_count` (shared across all 4 platforms), plus
  `city`/`state` (descriptive only, not a model input). Platform-specific:
  `story_post_rate_weekly` (IG), `retweet_ratio` (Twitter), `group_membership_count` (FB),
  `recommendation_count` + `connection_acceptance_rate` (LinkedIn). Sampled per-archetype,
  conditioned on `is_fake` via `common.TRUST_SIGNAL_DEFAULTS`/`sample_trust_signals()`, with
  optional per-archetype overrides on the `Archetype` dataclass -- added as *optional* fields
  (default `None`) specifically so none of the ~30 existing archetype definitions across 4
  platforms needed to be touched individually.
- All new numeric/boolean columns were added to `train_models.py`'s `BEHAVIORAL_BASE_COLS`
  as real model inputs, not just descriptive fields -- that was the actual point (more
  accurate Behavioral layer), not just a schema change.
- See `DATASET_GENERATION_SPEC.md`'s "Addendum (v2)" section for the authoritative column
  list.

**Scope tradeoff, stated explicitly**: given the phrase banks are ~1,400 lines of
hand-written content across 4 platforms, a full rewrite of every fragment with deep
Hinglish/regional flavor was not attempted -- the locale switch plus curated
company/city-state pools plus a light pass of festival/cricket/food fragments gets
genuine, evident Indian localization without that scope. Deeper linguistic localization
(code-switching, regional-language phrases) would be a reasonable next iteration if wanted.

## Production readiness pass

User asked to make the project production-ready. Scoped to local, additive, reversible
changes only -- no public deployment was performed (that needs the user's own hosting/
domain decision), and nothing here touches the existing scoring/lookup/history logic.

- **WSGI server**: `python app.py` now serves via `waitress` (cross-platform, works on
  Windows unlike gunicorn) instead of Flask's dev server. Same command, same default
  port (5000); override via `HOST`/`PORT` env vars for a container deployment.
- **Rate limiting** (Flask-Limiter, in-memory store): 30/min on `/api/lookup`, 60/min on
  history/saved writes, 20/min on clear-all, and a strict 5/min on `/api/scrape` (real
  external network calls to Instagram -- the most abuse-sensitive route, both for this
  app and to avoid getting the deploying IP flagged). Global default 200/hour elsewhere.
  Note: in-memory storage means limits are per-process, not shared across workers --
  fine for a single-process deploy (which `threads=os.cpu_count()` under waitress already
  parallelizes within one process); a true multi-process/multi-instance deployment would
  need a shared store (Redis) instead.
- **Security headers** (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`,
  `Permissions-Policy`) added via `after_request`. Deliberately did NOT add a strict CSP --
  the page loads Tailwind from a CDN and has an inline `<script>` injecting
  `window.SAMPLE_USERNAMES`/`window.PLATFORM_ICONS`; getting a CSP right for that without
  being able to live-test it risked silently breaking the page.
- **Logging**: rotating file handler (`logs/app.log`, 5MB x 3 backups) + console, for
  request-time errors/rate-limit events. Startup diagnostics still use `print()` on
  purpose (one-time human-readable status lines at import time).
- **`GET /healthz`**: liveness/readiness probe reporting whether models/semantic
  layer/DB/curves actually loaded, not just "the process is up." Exempt from rate
  limiting (orchestrators poll it frequently by design).
- **`MAX_CONTENT_LENGTH = 1MB`** on all requests, prevents large-payload abuse.
- **`requirements.txt`** rewritten from a handful of loosely-pinned lines (several core
  deps -- Flask, scikit-learn, xgboost, sentence-transformers, onnxruntime -- weren't even
  listed, having been installed ad hoc over the session) to a complete, exactly-pinned
  list matching what's actually installed and tested, so a fresh `pip install -r
  requirements.txt` reproduces this environment rather than drifting.
- **`Dockerfile` + `.dockerignore`**: non-root user, `libgomp1` (XGBoost's OpenMP
  runtime dependency), torch installed from PyTorch's CPU-only wheel index (avoids
  silently pulling a multi-GB CUDA build), `HEALTHCHECK` hitting `/healthz`, big
  training CSVs excluded from the build context (not needed at runtime, only by
  `train_models.py` offline) while the small `demo_*.csv` files are kept.

Also fixed while in this area: "Browse demo usernames" was silently capped at a 12-per-
platform sample (`SAMPLES_PER_LABEL = 6` genuine + 6 fake) despite a comment claiming it
browsed "the full set" -- it never did. `SAMPLE_USERNAMES` now includes every demo
profile (all ~200/platform, shuffled), so the expandable browse panel is a genuine browse
of the full ~800-profile demo set (per platform, via the existing platform switcher); the
compact landing-page chip row still shows only the first 6 (`static/app.js`,
`renderExampleChips`'s `.slice(0, 6)`), unchanged.

## Operational note for the next phase (feature extraction)

`hashtags` is stored as an empty string `""` for rows with no hashtags (Facebook/LinkedIn
mostly, per the spec's own note that this is expected). Plain `pd.read_csv(...)` will read
those empty cells back as `NaN`, not `""`, because pandas treats empty CSV fields as missing
by default. Load with `pd.read_csv(path, keep_default_na=False)` for that column, or
`.fillna("")` after loading, so downstream text vectorizers don't choke on `NaN`.
