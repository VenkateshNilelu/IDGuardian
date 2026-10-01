# IDGuardian — Synthetic Labeled Dataset Generation Spec

## Instructions for Claude Code

Build a Python script (or a small set of scripts, one per platform, plus a shared
`common.py` for shared helpers) that generates **synthetic, self-generated, no-real-data**
labeled datasets for IDGuardian's fake-profile detection project, for **four platforms**:
Instagram, Facebook, LinkedIn, and Twitter/X.

Do not scrape or use any real user data. Every bio, caption, username, and metric must be
programmatically generated (templates + randomized variation, e.g. via `Faker` and/or
hand-written phrase banks per archetype). Use a fixed random seed (e.g. `SEED = 42`) so the
dataset is reproducible.

For each platform, produce **two linked CSV files**: a Profile Dataset and a Post Dataset.
Save all eight output files under `./data/` in the project root, named:

```
data/instagram_profiles.csv   data/instagram_posts.csv
data/facebook_profiles.csv    data/facebook_posts.csv
data/linkedin_profiles.csv    data/linkedin_posts.csv
data/twitter_profiles.csv     data/twitter_posts.csv
```

**Default volume per platform (flag to the user if this should change before running at scale):**
5,000 profile records and 50,000 post records per platform (about 10 posts/profile on average,
not necessarily uniform -- sample post count per profile from a distribution centered on 10,
e.g. Poisson(10) clipped to >=1, and store the realized count in the profile's `posts_count`
field so it matches the actual number of linked posts).

---

## Addendum (v2): scale, India localization, platform-owner telemetry

Supersedes the volume line above and adds new columns. Applied in `common.py`/
`generate_dataset.py`; see `NOTES.md` for the full rationale.

- **Volume**: 20,000 profiles / ~200,000 posts per platform (4x v1), same ~10
  posts/profile distribution.
- **Locale**: Faker runs as `en_IN` (`common.FAKER_LOCALE`) -- every
  `{name}`/`{first}`/`{city}`/`{handle}` template fill is now Indian. Faker's
  `company()` under `en_IN` still leans Western-suffixed ("Kale LLC"), so
  `{company}` instead draws from a curated `INDIAN_COMPANY_NAMES` pool
  (Pvt Ltd / Technologies / Solutions style names). City/state are sampled
  together from a curated `INDIAN_CITY_STATE_PAIRS` list so they never
  mismatch (e.g. never "Mumbai, Sikkim"). New template placeholders
  `{festival}`/`{cricket}`/`{food}` (Diwali/Holi/IPL/chai/biryani, etc.) add
  cultural flavor to a subset of phrase-bank fragments (student/business/
  fitness banks).
- **New profile-level columns, shared across all 4 platforms** -- modeled as
  "platform owner" internal trust & safety telemetry: data a real platform's
  own systems would log, that no public scraping API ever exposes. Sampled
  per-archetype conditioned on `is_fake` (see `common.TRUST_SIGNAL_DEFAULTS` /
  `sample_trust_signals`), with per-archetype override support:
  - `city`, `state` -- descriptive only, not a model input (no discriminative
    signal in *which* Indian city a profile is in).
  - `email_verified`, `phone_verified`, `two_factor_enabled` (bool)
  - `device_count_30d`, `login_ip_diversity_30d` (int)
  - `signup_to_first_post_hours`, `posting_time_entropy` (float) -- bots skip
    the browse-before-posting delay real users show, and post on scripted,
    low-entropy schedules.
  - `follower_growth_rate_7d` (float) -- sudden spikes signal bought followers.
  - `reports_received_count`, `content_removed_count` (int) -- direct
    moderation history.
- **New platform-specific columns**:
  - Instagram: `story_post_rate_weekly`
  - Twitter/X: `retweet_ratio`
  - Facebook: `group_membership_count` (alongside existing `mutual_friends_count`)
  - LinkedIn: `recommendation_count`, `connection_acceptance_rate` (alongside
    existing `endorsements_count`/`skills_count`) -- written recommendations
    are far harder to fake at scale than one-click endorsements.
- All new numeric/boolean columns (excluding `city`/`state`) were added to
  `train_models.py`'s `BEHAVIORAL_BASE_COLS` as real Behavioral-layer model
  inputs, not just descriptive fields.

---

## 1. Cross-Cutting Rules (apply to all four platforms)

1. **Linkage.** Every row in a Post Dataset must carry a `user_id` that references an
   existing row in that platform's Profile Dataset. Never generate an orphan post.

2. **Label consistency.** A post's `archetype` and `is_fake` must always equal its parent
   profile's `archetype` and `is_fake`. Never let a post disagree with its own profile's
   label -- this would inject label noise.

3. **Aggregate consistency.** Profile-level `avg_likes`, `avg_comments`, `avg_shares`, and
   `engagement_rate` must be computed as the actual mean of that profile's linked posts in
   the Post Dataset -- do NOT sample these independently at the profile level. Generate the
   posts first, then derive the profile-level aggregates from them.

4. **Balanced-but-realistic labels.** Aim for roughly 60% genuine / 40% fake overall per
   platform (not a strict 50/50 -- real-world fake-account prevalence is lower than genuine,
   but keep enough fake examples for the classifiers to learn from). Distribute the fake
   share across that platform's fake archetypes roughly evenly, and the genuine share across
   that platform's genuine archetypes roughly evenly, unless noted otherwise below.

5. **Archetype-conditioned generation, not just archetype-conditioned labels.** Every
   numeric and text field should be sampled from a distribution/template bank that depends
   on the row's archetype, not sampled independently of it. E.g.: fake "Spam Account" bios
   should be templated/promotional; genuine "Student" bios should be casual, varied, and
   contain personal details (school-adjacent hobbies, non-promotional tone); fake accounts
   generally skew toward younger `account_age_days`, near-1.0 follower/following-style
   ratios (where applicable), and low genuine engagement relative to their audience size;
   genuine accounts skew toward older accounts, more organic-looking engagement.

6. **No leakage columns.** Do not include any column that trivially encodes `is_fake` in an
   unrealistic way (e.g. a literal "fake_score" field). The label must be inferable only
   from the same kind of signals a real system would see (text, counts, ratios), so the
   downstream lexical/semantic/behavioral models have something real to learn.

7. **Text generation.** Bios, headlines, and captions should be generated from
   archetype-specific phrase banks with randomized recombination (not the exact same string
   repeated verbatim across all rows of that archetype -- vary word choice/order/emoji so
   TF-IDF and SBERT see realistic diversity, not one memorized template).

8. **Sentiment field.** Compute `sentiment` (post dataset) programmatically from the
   generated caption text using a simple, deterministic sentiment scorer (e.g. VADER or a
   small lexicon-based scorer) rather than hand-labeling it -- keep it consistent with the
   actual generated text.

9. **Validation pass.** At the end of the script, run and print a validation report:
   - No missing `user_id` references between posts and profiles.
   - No post/profile label mismatches.
   - Profile-level aggregates match computed post-level means within a small tolerance.
   - Per-archetype row counts and overall fake/genuine balance, per platform.
   - Basic schema check (correct columns, correct dtypes, no unexpected nulls outside the
     documented exceptions below).

---

## 2. Instagram

**Archetypes** -- Genuine: `Student`, `Developer`, `Business Account`, `Fitness Creator`.
Fake: `Spam Account`, `Fake Influencer`, `Celebrity Impersonator`.

### instagram_profiles.csv

| Column | Type | Notes |
|---|---|---|
| user_id | string | unique |
| username | string | |
| bio | string | archetype-conditioned |
| followers_count | int | |
| following_count | int | |
| posts_count | int | must equal linked post count |
| account_age_days | int | fake skews younger |
| avg_likes | float | derived from linked posts |
| avg_comments | float | derived from linked posts |
| avg_shares | float | derived from linked posts |
| engagement_rate | float | derived from linked posts |
| is_verified | bool | |
| has_website | bool | |
| has_location | bool | |
| profile_completion_score | float 0-1 | |
| archetype | string | one of the 7 above |
| is_fake | int (0/1) | |

**Derived features (compute in a preprocessing step, not stored as raw columns unless you
want them precomputed for convenience):** `follow_ratio = followers_count / (following_count + 1)`,
`follower_gap = followers_count - following_count`.

### instagram_posts.csv
`user_id, caption, hashtags, likes, comments, shares, sentiment, archetype, is_fake`

---

## 3. Facebook

**Archetypes** -- Genuine: `Student`, `Developer`, `Business Page`, `Fitness Creator`. Fake:
`Spam Account`, `Fake Influencer`, `Celebrity Impersonator`, `Catfish/Romance-Scam Profile`.

### facebook_profiles.csv

| Column | Type | Notes |
|---|---|---|
| user_id | string | unique |
| username | string | |
| bio | string | "About" text |
| friends_count | int | personal-profile connection count |
| followers_count | int | Page/public-follow count (can be 0 for personal profiles) |
| posts_count | int | must equal linked post count |
| account_age_days | int | |
| avg_likes | float | derived |
| avg_comments | float | derived |
| avg_shares | float | derived |
| engagement_rate | float | derived |
| mutual_friends_count | int | Facebook-specific behavioral signal; low for fake/catfish |
| is_verified | bool | |
| has_website | bool | |
| has_location | bool | |
| profile_completion_score | float 0-1 | |
| archetype | string | |
| is_fake | int (0/1) | |

### facebook_posts.csv
`user_id, caption, hashtags, likes, comments, shares, sentiment, archetype, is_fake`
(hashtags may often be empty string -- that's expected for Facebook.)

---

## 4. LinkedIn

**Archetypes** -- Genuine: `Student`, `Developer`, `Business/Company Page`, `Consultant`.
Fake: `Fake Recruiter`, `Fake Executive Impersonator`, `Spam/Bot Account`.

LinkedIn connections are mutual -- **do not generate a separate `following_count`**, and do
**not** compute `follow_ratio`/`follower_gap` for this platform (there is no equivalent).
Use `connections_count`, `endorsements_count`, and `skills_count` as this platform's
engineered behavioral signals instead.

### linkedin_profiles.csv

| Column | Type | Notes |
|---|---|---|
| user_id | string | unique |
| username | string | |
| headline | string | bio equivalent |
| connections_count | int | mutual, capped realistically (LinkedIn caps at 30,000) |
| posts_count | int | must equal linked post count |
| account_age_days | int | |
| avg_likes | float | derived ("reactions") |
| avg_comments | float | derived |
| avg_shares | float | derived ("reposts") |
| engagement_rate | float | derived |
| job_title | string | fake recruiter/exec archetypes: generic/inconsistent titles |
| company_name | string | fake archetypes: implausible or mismatched company |
| endorsements_count | int | |
| skills_count | int | |
| is_verified | bool | |
| has_website | bool | |
| has_location | bool | |
| profile_completion_score | float 0-1 | |
| archetype | string | |
| is_fake | int (0/1) | |

### linkedin_posts.csv
`user_id, caption, hashtags, likes, comments, shares, sentiment, archetype, is_fake`
(hashtags will be sparse/often empty -- expected for LinkedIn.)

---

## 5. Twitter / X

**Archetypes** -- Genuine: `Student`, `Developer`, `Business Account`, `Fitness Creator`.
Fake: `Spam/Bot Account`, `Fake Influencer`, `Celebrity Impersonator`.

### twitter_profiles.csv

| Column | Type | Notes |
|---|---|---|
| user_id | string | unique |
| username | string | |
| bio | string | |
| followers_count | int | |
| following_count | int | |
| tweets_count | int | must equal linked post count (this platform's "posts_count") |
| account_age_days | int | |
| avg_likes | float | derived |
| avg_comments | float | derived ("replies") |
| avg_shares | float | derived ("retweets") |
| engagement_rate | float | derived |
| is_verified | bool | |
| has_website | bool | |
| has_location | bool | |
| profile_completion_score | float 0-1 | |
| archetype | string | |
| is_fake | int (0/1) | |

**Derived features:** `follow_ratio = followers_count / (following_count + 1)`,
`follower_gap = followers_count - following_count` (same as Instagram -- Twitter has the
same follow/following asymmetry).

### twitter_posts.csv
`user_id, caption, hashtags, likes, comments, shares, sentiment, archetype, is_fake`

---

## 6. Deliverables Checklist

- [ ] `data/` folder with the 8 CSV files listed above.
- [ ] Generation script(s) checked into the repo (not just run once and discarded) so the
      dataset is regenerable and auditable.
- [ ] A printed/saved validation report (as described in Cross-Cutting Rule 9) -- save it as
      `data/validation_report.txt` alongside the CSVs.
- [ ] No real scraped data anywhere in the pipeline -- confirm every field is synthetically
      generated.

## 7. Open Items to Confirm With the User Before Finalizing

- Confirm the default volume (5,000 profiles / 50,000 posts **per platform**) is correct,
  or adjust if a different total/split is wanted.
- Confirm the genuine/fake balance target (default proposed here: 60/40).
- Facebook's `Catfish/Romance-Scam Profile` and LinkedIn's `Fake Recruiter` /
  `Fake Executive Impersonator` archetypes were added here as platform-appropriate fake
  patterns beyond the original Instagram-only archetype list -- confirm these are wanted, or
  simplify to match the original three fake archetypes (Spam Account, Fake Influencer,
  Celebrity Impersonator) on every platform for consistency.

---

## 8. Recommended Model Architecture for Next Phase (not part of dataset generation — for later reference)

Once the 8 CSVs above are generated and validated, the recommended training split for the
IDGuardian four-layer model across platforms is:

- **Lexical Layer** (TF-IDF + Logistic Regression) — train ONE shared model on the pooled
  text corpus (bio/headline + captions + hashtags) across all four platforms.
- **Semantic Layer** (Sentence-BERT + XGBoost) — train ONE shared model on pooled SBERT
  embeddings across all four platforms.
- **Behavioral Layer** (XGBoost) — train FOUR separate models, one per platform, since the
  feature sets are structurally different (e.g. LinkedIn has no follow_ratio/follower_gap).
- **Trust Fusion Layer** (Logistic Regression) — train FOUR separate models, one per
  platform, since each base layer's reliability differs by platform and fusion is cheap
  enough to not need sharing.

Net: 2 shared models + 8 platform-specific models (Behavioral x4, Fusion x4) = 10 trained
models total. Algorithms 1-4 and the Mathematical Model already written for the paper do not
need to change structurally — only the Methodology section needs one added paragraph
explaining which layers are pooled vs. per-platform.
