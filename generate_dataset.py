"""
IDGuardian synthetic dataset generator.

Generates labeled Profile + Post datasets for Instagram, Facebook, LinkedIn,
and Twitter/X per DATASET_GENERATION_SPEC.md. Every field is synthetically
generated (Faker + hand-written archetype phrase banks) -- no real scraped
data anywhere. Run with a fixed SEED (see common.py) for reproducibility.

Usage:
    python generate_dataset.py                 # generate all four platforms
    python generate_dataset.py --platform ig    # generate just one
                                                 # (ig|fb|li|tw)
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

import common as C

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ===========================================================================
# Archetype configuration
# ===========================================================================

@dataclass
class Archetype:
    name: str
    is_fake: int
    text_bank: C.PhraseBank            # bio / headline
    caption_bank: C.CaptionBank
    hashtag_pool: list
    username_style: str = "casual"     # casual | spammy | aesthetic | impersonator | corporate
    hashtag_range: tuple = (1, 4)
    hashtag_empty_prob: float = 0.05
    account_age_range: tuple = (200, 3000)
    primary_range: tuple = (100, 5000)         # main audience metric (followers/connections/friends)
    secondary_mode: str = "none"               # none | ratio_near_1 | independent
    secondary_range: tuple = (0, 0)            # used when independent
    ratio_range: tuple = (0.8, 1.3)            # used when ratio_near_1 (secondary = primary * ratio)
    engagement_rate_range: tuple = (0.02, 0.08)
    comment_ratio_range: tuple = (0.05, 0.2)
    share_ratio_range: tuple = (0.01, 0.06)
    post_noise_sigma_range: tuple = (0.4, 0.7)
    is_verified_p: float = 0.02
    has_website_p: float = 0.2
    has_location_p: float = 0.4
    profile_completion_range: tuple = (0.5, 0.9)
    # Platform-owner trust & safety telemetry -- optional per-archetype overrides;
    # None means "use the is_fake-conditioned default" (see common.TRUST_SIGNAL_DEFAULTS).
    email_verified_p: Optional[float] = None
    phone_verified_p: Optional[float] = None
    two_factor_p: Optional[float] = None
    device_count_range: Optional[tuple] = None
    ip_diversity_range: Optional[tuple] = None
    signup_to_post_range: Optional[tuple] = None
    posting_entropy_range: Optional[tuple] = None
    follower_growth_range: Optional[tuple] = None
    reports_range: Optional[tuple] = None
    content_removed_range: Optional[tuple] = None


def make_username(rng: np.random.Generator, fk, archetype: Archetype) -> str:
    base = fk.user_name()
    style = archetype.username_style
    if style == "casual":
        if rng.random() < 0.3:
            base = f"{base}{int(rng.integers(1, 999))}"
        return base
    if style == "corporate":
        return fk.company().lower().replace(",", "").replace(".", "").replace(" ", "_")[:24]
    if style == "spammy":
        prefixes = ["get", "best", "top", "real", "official", "promo", "shop", "win", "free", "the"]
        p = prefixes[int(rng.integers(0, len(prefixes)))]
        digits = int(rng.integers(100, 99999))
        return f"{p}_{base}{digits}"
    if style == "aesthetic":
        seps = ["_", ".", ""]
        sep = seps[int(rng.integers(0, len(seps)))]
        suffix = ["official", "vip", "world", "life", "xo", "glow", "co"][int(rng.integers(0, 7))]
        return f"{base}{sep}{suffix}"
    if style == "impersonator":
        suffix = ["_real", "_official", ".official", "_fanpage", "_verified", "1", "_hq"][int(rng.integers(0, 7))]
        return f"{base}{suffix}"
    return base


def even_split(total: int, k: int) -> list[int]:
    if k <= 0:
        return []
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


ExtraFieldBuilder = Callable[[np.random.Generator, object, int], dict[str, np.ndarray]]


def sample_numeric_fields(rng: np.random.Generator, source: Archetype, k: int) -> dict:
    """Sample every archetype-conditioned numeric profile field from `source`'s config."""
    age = C.sample_uniform_int(rng, *source.account_age_range, k)
    primary = np.round(C.sample_lognormal_clipped(rng, *source.primary_range, k)).astype(int)
    if source.secondary_mode == "ratio_near_1":
        mult = C.sample_uniform_float(rng, *source.ratio_range, k)
        secondary = np.round(primary * mult).clip(min=1).astype(int)
    elif source.secondary_mode == "independent":
        secondary = np.round(C.sample_lognormal_clipped(rng, *source.secondary_range, k)).astype(int)
    else:
        secondary = np.zeros(k, dtype=int)
    trust = C.sample_trust_signals(rng, source.is_fake, k, overrides=dict(
        email_verified_p=source.email_verified_p, phone_verified_p=source.phone_verified_p,
        two_factor_p=source.two_factor_p, device_count_range=source.device_count_range,
        ip_diversity_range=source.ip_diversity_range, signup_to_post_range=source.signup_to_post_range,
        posting_entropy_range=source.posting_entropy_range, follower_growth_range=source.follower_growth_range,
        reports_range=source.reports_range, content_removed_range=source.content_removed_range,
    ))
    return dict(
        age=age, primary=primary, secondary=secondary,
        eng_base=C.sample_uniform_float(rng, *source.engagement_rate_range, k),
        comment_ratio=C.sample_uniform_float(rng, *source.comment_ratio_range, k),
        share_ratio=C.sample_uniform_float(rng, *source.share_ratio_range, k),
        completion=np.round(C.sample_uniform_float(rng, *source.profile_completion_range, k), 3),
        verified=C.sample_bool(rng, source.is_verified_p, k),
        website=C.sample_bool(rng, source.has_website_p, k),
        location=C.sample_bool(rng, source.has_location_p, k),
        **trust,
    )


def assign_shadow_sources(rng: np.random.Generator, arche: Archetype, cnt: int,
                           archetypes: list[Archetype], frac: float) -> list[Archetype]:
    """For `frac` of `cnt` rows, swap in a randomly chosen opposite-class archetype as
    that row's "source" for whatever the caller samples from next (numeric fields, or
    text banks) -- the row keeps its own true archetype/is_fake label regardless."""
    sources = [arche] * cnt
    opposite_pool = [a for a in archetypes if a.is_fake != arche.is_fake]
    if frac > 0 and opposite_pool:
        n_ov = int(round(cnt * frac))
        if n_ov > 0:
            ov_idx = rng.choice(cnt, size=n_ov, replace=False)
            shadow_pick = rng.integers(0, len(opposite_pool), size=n_ov)
            for pos, row_i in enumerate(ov_idx):
                sources[row_i] = opposite_pool[shadow_pick[pos]]
    return sources


def build_platform_dataset(
    platform: str,
    prefix: str,
    archetypes: list[Archetype],
    n_profiles: int,
    posts_count_field: str,
    primary_field: str,
    secondary_field: Optional[str],
    engagement_denominator_fields: list[str],
    extra_field_builders: dict[str, ExtraFieldBuilder],
    profile_column_order: list[str],
    text_field: str,
    overlap_frac: float = 0.0,
    text_overlap_frac: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    `overlap_frac`: for this share of EVERY archetype's profiles, the ENTIRE numeric
    fingerprint (account age, audience size, engagement rate, comment/share ratios,
    profile completion, verified/website/location) is redrawn from a randomly chosen
    *opposite-class* archetype's config instead of the profile's own -- i.e. some fake
    accounts behave like a genuine archetype end-to-end, and vice versa, while keeping
    their true archetype/is_fake label.

    This has to touch every numeric field together, not just one or two: an "ambiguous"
    subset with noise on only account_age_days/engagement_rate still leaves every other
    field intact, so a nonlinear classifier can just fingerprint which archetype cluster a
    row belongs to from the untouched fields and back out is_fake from that (archetype
    determines is_fake deterministically by construction). Randomizing the whole numeric
    vector together for the overlapped rows is what actually caps a combined classifier's
    achievable accuracy, rather than merely denting one or two univariate signals.

    `text_overlap_frac`: the same idea applied independently to bio/headline + captions +
    hashtags (drawn from a shadow archetype's *text banks* instead of its numeric config).
    This needs its own, generally much larger, fraction: each archetype's phrase banks use
    essentially disjoint vocabulary from every other archetype's (a spam bank never uses a
    genuine bank's words), so a bag-of-words/embedding classifier can key off a handful of
    highly distinctive tokens and reach ~100% accuracy even when the numeric layer alone is
    realistically noisy -- discovered when the trained Lexical/Semantic layers came back at
    99.9-100% test accuracy on every platform despite the tuned 86-93% Behavioral layer,
    which would make Trust Fusion a no-op that adds nothing over Lexical alone.
    """
    rng = C.make_rng(platform)
    fk = C.make_faker(platform)

    genuine = [a for a in archetypes if a.is_fake == 0]
    fake = [a for a in archetypes if a.is_fake == 1]
    n_genuine = round(n_profiles * C.GENUINE_SHARE)
    n_fake = n_profiles - n_genuine
    genuine_counts = even_split(n_genuine, len(genuine))
    fake_counts = even_split(n_fake, len(fake))
    blocks = list(zip(genuine, genuine_counts)) + list(zip(fake, fake_counts))

    TRUST_COLS = ["email_verified", "phone_verified", "two_factor_enabled", "device_count_30d",
                  "login_ip_diversity_30d", "signup_to_first_post_hours", "posting_time_entropy",
                  "follower_growth_rate_7d", "reports_received_count", "content_removed_count"]
    cols: dict[str, list] = {
        "text": [], "username": [], "archetype": [], "is_fake": [], "text_source": [],
        "account_age_days": [], "is_verified": [], "has_website": [], "has_location": [],
        "profile_completion_score": [], "primary": [], "secondary": [],
        "engagement_rate_base": [], "comment_ratio": [], "share_ratio": [],
        "post_noise_sigma": [], "n_posts": [],
        **{c: [] for c in TRUST_COLS},
    }
    extra_accum: dict[str, list[np.ndarray]] = {}
    caption_banks: dict[str, C.CaptionBank] = {}
    hashtag_cfg: dict[str, tuple] = {}

    for arche, cnt in blocks:
        caption_banks[arche.name] = arche.caption_bank
        hashtag_cfg[arche.name] = (arche.hashtag_pool, *arche.hashtag_range, arche.hashtag_empty_prob)
        if cnt <= 0:
            continue

        # Independent shadow assignments for text vs. numeric fields -- text needs a much
        # larger overlap fraction than numeric to land in a realistic accuracy range (see
        # docstring), so they're tuned and drawn separately rather than sharing one array.
        text_sources = assign_shadow_sources(rng, arche, cnt, archetypes, text_overlap_frac)
        sources = assign_shadow_sources(rng, arche, cnt, archetypes, overlap_frac)

        cols["text"].extend(text_sources[i].text_bank.sample(rng, fk) for i in range(cnt))
        cols["text_source"].extend(s.name for s in text_sources)
        cols["username"].extend(make_username(rng, fk, arche) for _ in range(cnt))
        cols["archetype"].extend([arche.name] * cnt)
        cols["is_fake"].extend([arche.is_fake] * cnt)

        age = np.empty(cnt, dtype=int)
        primary = np.empty(cnt, dtype=int)
        secondary = np.empty(cnt, dtype=int)
        eng_base = np.empty(cnt, dtype=float)
        comment_ratio = np.empty(cnt, dtype=float)
        share_ratio = np.empty(cnt, dtype=float)
        completion = np.empty(cnt, dtype=float)
        verified = np.empty(cnt, dtype=bool)
        website = np.empty(cnt, dtype=bool)
        location = np.empty(cnt, dtype=bool)
        trust_arrs = {
            c: np.empty(cnt, dtype=bool if c in ("email_verified", "phone_verified", "two_factor_enabled")
                        else float if c in ("signup_to_first_post_hours", "posting_time_entropy",
                                             "follower_growth_rate_7d") else int)
            for c in TRUST_COLS
        }

        for src in {id(s): s for s in sources}.values():
            idxs = np.array([i for i in range(cnt) if sources[i] is src])
            vals = sample_numeric_fields(rng, src, len(idxs))
            age[idxs] = vals["age"]
            primary[idxs] = vals["primary"]
            secondary[idxs] = vals["secondary"]
            eng_base[idxs] = vals["eng_base"]
            comment_ratio[idxs] = vals["comment_ratio"]
            share_ratio[idxs] = vals["share_ratio"]
            completion[idxs] = vals["completion"]
            verified[idxs] = vals["verified"]
            website[idxs] = vals["website"]
            location[idxs] = vals["location"]
            for c in TRUST_COLS:
                trust_arrs[c][idxs] = vals[c]

        cols["account_age_days"].extend(age.tolist())
        cols["is_verified"].extend(verified.tolist())
        cols["has_website"].extend(website.tolist())
        cols["has_location"].extend(location.tolist())
        cols["profile_completion_score"].extend(completion.tolist())
        cols["primary"].extend(primary.tolist())
        cols["secondary"].extend(secondary.tolist())
        cols["engagement_rate_base"].extend(eng_base.tolist())
        cols["comment_ratio"].extend(comment_ratio.tolist())
        cols["share_ratio"].extend(share_ratio.tolist())
        cols["post_noise_sigma"].extend(C.sample_uniform_float(rng, *arche.post_noise_sigma_range, cnt).tolist())
        cols["n_posts"].extend(C.sample_poisson_min1(rng, C.POSTS_PER_PROFILE_LAMBDA, cnt).tolist())
        for c in TRUST_COLS:
            cols[c].extend(trust_arrs[c].tolist())

        if arche.name in extra_field_builders:
            extras = extra_field_builders[arche.name](rng, fk, cnt)
            for k, v in extras.items():
                extra_accum.setdefault(k, []).append(np.asarray(v))

    n_total = len(cols["text"])
    user_ids = np.array(C.make_user_ids(prefix, n_total))
    perm = rng.permutation(n_total)

    def P(key):
        return np.array(cols[key], dtype=object)[perm]

    def Pnum(key, dtype):
        return np.array(cols[key])[perm].astype(dtype)

    archetype_arr = P("archetype")
    text_source_arr = P("text_source")
    is_fake_arr = Pnum("is_fake", "int64")
    n_posts_arr = Pnum("n_posts", "int64")
    primary_arr = Pnum("primary", "int64")
    secondary_arr = Pnum("secondary", "int64")

    extra_final: dict[str, np.ndarray] = {}
    for k, chunks in extra_accum.items():
        full = np.concatenate(chunks)
        extra_final[k] = full[perm]

    # ---- posts ----
    # The audience used to drive post-level engagement must match the
    # denominator used later for engagement_rate, or archetypes whose reach
    # lives mostly in the secondary count (e.g. Facebook Pages, where
    # followers_count dwarfs friends_count) would show absurdly deflated
    # engagement_rate despite realistic-looking post engagement.
    audience_arr = primary_arr.astype(float)
    if secondary_field is not None and secondary_field in engagement_denominator_fields:
        audience_arr = audience_arr + secondary_arr.astype(float)

    posts = C.generate_posts_for_all_profiles(
        rng, fk, user_ids.tolist(), archetype_arr.tolist(), is_fake_arr.tolist(), n_posts_arr.tolist(),
        audience=audience_arr.tolist(),
        engagement_rate_base=Pnum("engagement_rate_base", "float64").tolist(),
        comment_ratio=Pnum("comment_ratio", "float64").tolist(),
        share_ratio=Pnum("share_ratio", "float64").tolist(),
        post_noise_sigma=Pnum("post_noise_sigma", "float64").tolist(),
        caption_banks=caption_banks,
        hashtag_cfg=hashtag_cfg,
        text_sources=text_source_arr.tolist(),
    )

    agg = C.derive_profile_aggregates(posts, user_ids.tolist())

    cities, states = C.sample_city_state(rng, n_total)
    profile_data = {
        "user_id": user_ids,
        "username": P("username"),
        primary_field: primary_arr,
        "account_age_days": Pnum("account_age_days", "int64"),
        "is_verified": Pnum("is_verified", "bool"),
        "has_website": Pnum("has_website", "bool"),
        "has_location": Pnum("has_location", "bool"),
        "profile_completion_score": Pnum("profile_completion_score", "float64"),
        "city": np.array(cities), "state": np.array(states),
        "email_verified": Pnum("email_verified", "bool"),
        "phone_verified": Pnum("phone_verified", "bool"),
        "two_factor_enabled": Pnum("two_factor_enabled", "bool"),
        "device_count_30d": Pnum("device_count_30d", "int64"),
        "login_ip_diversity_30d": Pnum("login_ip_diversity_30d", "int64"),
        "signup_to_first_post_hours": Pnum("signup_to_first_post_hours", "float64"),
        "posting_time_entropy": Pnum("posting_time_entropy", "float64"),
        "follower_growth_rate_7d": Pnum("follower_growth_rate_7d", "float64"),
        "reports_received_count": Pnum("reports_received_count", "int64"),
        "content_removed_count": Pnum("content_removed_count", "int64"),
        "archetype": archetype_arr,
        "is_fake": is_fake_arr,
    }
    if secondary_field:
        profile_data[secondary_field] = secondary_arr
    profile_data.update(extra_final)

    profiles = pd.DataFrame(profile_data)
    profiles[text_field] = P("text")

    profiles = profiles.merge(agg, on="user_id", how="left")
    profiles = profiles.rename(columns={"posts_count": posts_count_field})

    denom = np.zeros(len(profiles))
    for f in engagement_denominator_fields:
        denom = denom + profiles[f].to_numpy(dtype=float)
    denom = np.clip(denom, 1, None)
    profiles["engagement_rate"] = (
        profiles["avg_likes"] + profiles["avg_comments"] + profiles["avg_shares"]
    ) / denom

    profiles = profiles[profile_column_order]
    return profiles, posts


# ===========================================================================
# Shared phrase-bank building blocks
# ===========================================================================

EMOJI_CASUAL = ["😊", "✨", "🙌", "📚", "🎉", "💪", "🌟", "☕", "🐾", "🎬"]
EMOJI_DEV = ["💻", "⚙️", "🚀", "🐛", "🔧", "☕", "📦", "🧠"]
EMOJI_BIZ = ["📈", "🤝", "💼", "🌍", "✅", "🔗"]
EMOJI_FIT = ["💪", "🔥", "🏋️", "🥗", "🏃", "⚡", "🧘"]
EMOJI_SPAM = ["🔥", "💰", "🎁", "👉", "✅", "💯", "📩", "🛒"]
EMOJI_GLAM = ["✨", "💖", "😍", "💅", "🌴", "📸", "👑"]

HASHTAGS_STUDENT = ["studentlife", "college", "campus", "studygram", "gradstudent", "classof2026",
                     "dormlife", "finals", "studybreak", "university"]
HASHTAGS_DEV = ["coding", "developer", "softwareengineer", "opensource", "webdev", "python",
                "javascript", "buildinpublic", "100daysofcode", "programmer"]
HASHTAGS_BIZ = ["smallbusiness", "entrepreneur", "startup", "localbusiness", "customerfirst",
                "businessowner", "growth", "marketing", "shopsmall", "hustle"]
HASHTAGS_FIT = ["fitness", "gymlife", "workout", "fitfam", "nutrition", "healthylifestyle",
                "training", "strength", "cardio", "wellness"]
HASHTAGS_SPAM = ["makemoneyonline", "followforfollow", "giveaway", "cryptosignals", "dmforprice",
                 "cashapp", "onlyfollowers", "getrichquick", "freeprize", "clicklink"]
HASHTAGS_INFLUENCER = ["ad", "sponsored", "influencer", "brandpartner", "collab", "linkinbio",
                        "lifestyle", "trending", "musthave", "discountcode"]
HASHTAGS_CELEB = ["fanpage", "celebrity", "trending", "viral", "exclusive", "official"]

# High-cardinality "detail" clauses: almost every bio/caption gets one of these,
# and most of them carry a Faker/number placeholder. A handful of fixed phrase
# fragments alone produce only a few hundred distinct combinations, which
# collapses into heavy duplication once an archetype has thousands of rows --
# a 4-digit number or a Faker city/name multiplies the space by 1-2 orders of
# magnitude and is what actually keeps exact-duplicate rates low at scale.
DET_STUDENT = [
    "Currently {number} cups of coffee deep today.", "Fun fact: I've switched majors {number} times.",
    "Shoutout to {first} for the study session.", "Somewhere in {city}, probably procrastinating.",
    "This is assignment number {number} this semester.", "Currently {number} days away from break.",
    "Tagging {first} because they'd get this.", "Can't believe it's already {year}.",
    "Number {number} on my to-do list: sleep.", "Group chat with {first} is unhinged today.",
    "Campus tour {number} for prospective students today.", "Officially {number} credits away from graduating.",
    "Studying in {city} has its perks.", "Ran into {first} at the library again.",
]
DET_DEV = [
    "Deploy number {number} of the week.", "Shoutout to {first} for the code review.",
    "Currently debugging from {city}.", "This is bug ticket number {number}, send help.",
    "{number} tabs of documentation open right now.", "Working out of {city} this {year}.",
    "PR number {number} finally merged.", "Tagging {first}, they'll appreciate this one.",
    "Still can't believe it's {year} and this bug exists.", "Sprint {number} wrapped up today.",
    "Standup update: {number} things fixed today.", "Currently on commit number {number} of this refactor.",
    "Pairing with {first} made this so much faster.", "Working remotely from {city} this month.",
]
DET_BIZ = [
    "Serving {city} since {year}.", "Order number {number} just shipped.",
    "Proudly {number} years in business this {year}.", "Shoutout to our {number}th customer this week.",
    "Based out of {city}, shipping everywhere.", "Team huddle today covered {number} new ideas.",
    "Customer number {number} just left a five-star review.", "Thank you {city} for {number} years of support.",
    "Batch number {number} restocked today.", "{number} orders packed and ready to go.",
    "Celebrating {number} years since we opened our doors.", "Tagging {first}, one of our first customers back in {year}.",
]
DET_FIT = [
    "Day {number} of the program.", "Client hit a {number} pound PR today.",
    "{number} reps, zero regrets.", "Training out of {city} this week.",
    "Week {number} of the plan, feeling stronger.", "Shoutout to {first} for showing up every day.",
    "{number} minutes of mobility work this morning.", "Session number {number} with this client, big progress.",
    "Coaching {number} clients this cycle.", "Back in {city} for a pop-up class this {year}.",
    "Logged {number} miles this week.", "Tagging {first}, my training partner today.",
]
DET_SPAM = [
    "Over {number} people already joined this week.", "Spot number {number} just opened up.",
    "This worked for {first} too, ask them.", "{number} people can't all be wrong.",
    "Started in {city}, now everywhere.", "Already {number} success stories this month.",
    "Slot {number} almost gone.", "{first} just signed up, you're next.",
    "Since {year}, thousands have joined.", "Only {number} spots left today.",
    "Already helped {number} people this {year}.", "Message number {number} today, still going.",
]
DET_INFLUENCER = [
    "Shot this in {city}.", "Outfit number {number} of the week.", "{first} styled this look.",
    "Can't believe it's already {year}.", "Trip number {number} this year, no complaints.",
    "Brand deal number {number} this month.", "Filming this in {city} today.",
    "So many of you asked for this, here it is.", "Tagging {first}, my glam team today.",
    "Look number {number} from this campaign.", "Since {year}, this has been the dream.",
    "Packing for {city} again this week.",
]
DET_IMPERSONATOR = [
    "Since {year}, still going strong.", "{number} years in this industry now.",
    "Thank you {first} for the constant support.", "Project number {number} coming together nicely.",
    "Back in {city} this {year}.", "{number} years of your support means everything.",
    "Working on something big for {year}.", "Message number {number} today, reading every one.",
    "Since {year}, the journey continues.", "Tagging {first}, thank you for believing in this.",
]
DET_CATFISH = [
    "Currently stationed near {city}.", "Hoping to visit {city} again someday.",
    "It's been {number} months since we last spoke.", "Thinking of you from {city} tonight.",
    "Only {number} more months until I'm back.", "Since {year}, I've been looking for this.",
    "{number} letters written, still waiting to send them.", "Missing {city} and missing you more.",
]
DET_LI_STUDENT = [
    "Cohort of {year}, {city} campus.", "Project number {number} this semester.",
    "Shoutout to {first} for the mentorship.", "{number} credits away from graduating.",
    "Career fair number {number} this year.", "Internship application number {number} submitted.",
    "Class of {year}, still learning every day.", "Met with a career advisor in {city} this week.",
    "Grateful for {first}'s advice on this one.", "One of {number} students selected for the program.",
]
DET_LI_DEV = [
    "Shipped feature number {number} this sprint.", "Sprint {number} retro done.",
    "Mentoring {number} junior engineers this quarter.", "PR number {number} merged today.",
    "Based out of {city}, remote-first.", "{number} years into this role now.",
    "Shoutout to {first} for the pairing session.", "Working with the team at {company} on this one.",
    "Spoke with {first} on the platform team about it.", "Rolling this out from our {city} office.",
]
DET_LI_BIZ = [
    "Based in {city}, serving clients since {year}.", "Milestone number {number} unlocked this quarter.",
    "Proudly {number} years in business.", "Welcoming {number} new team members this {year}.",
    "Thank you to our {number}th client.", "Expanding into {city} this {year}.",
    "Congrats to {first} on leading this launch.", "Opening a new office in {city} this {year}.",
    "Partnering with {company} on this initiative.",
]
DET_LI_CONSULTANT = [
    "Engagement number {number} wrapped this quarter.", "{number} years advising clients across industries.",
    "Based out of {city} this {year}.", "Case study number {number} published this month.",
    "Speaking at {number} conferences this {year}.", "Worked alongside {first} on this engagement.",
    "Advising a team at {company} this quarter.", "Presenting findings in {city} next month.",
]
DET_LI_RECRUITER = [
    "Filled {number} roles this month alone.", "Position number {number} still open, apply now.",
    "{number} candidates placed since {year}.", "Hiring event in {city} this week.",
    "Slot number {number} still available.", "Working with {company} to fill this fast.",
    "Ask for {first}, I'm handling this search.", "Recruiting drive kicking off in {city} this week.",
]
DET_LI_EXEC = [
    "Since {year}, leading this company forward.", "{number} years of industry-leading growth.",
    "Announcement number {number} this quarter.", "Based in {city}, thinking globally.",
    "Milestone number {number} for the company this {year}.", "Meeting with {first} to plan the next phase.",
    "Expanding {company} into {city} this {year}.", "Addressed the board on this in {city} last week.",
]
DET_LI_SPAM = [
    "Helped {number} clients hit six figures this {year}.", "Session number {number} of my mentorship program.",
    "Since {year}, thousands have joined my program.", "Only {number} spots left this month.",
    "Case study number {number} of my system, ask me how.", "Ask {first}, they joined last month.",
    "Ran a session in {city} last week, huge turnout.", "Built this system while working out of {city}.",
]


def bank_student() -> C.PhraseBank:
    return C.PhraseBank(
        fragments={
            "role": ["{first} | student", "Junior at {school}", "Studying at {school}", "{school} '27",
                     "Full-time student, part-time dreamer", "Grad student @ {school}"],
            "study": ["majoring in something I still can't explain at parties", "CS major, coffee minor",
                      "trying to survive finals week", "psych major who overthinks everything",
                      "business major, meme connoisseur", "pre-med and perpetually tired"],
            "hobby": ["obsessed with true crime podcasts", "into indie films and bad puns",
                      "weekend hiker, weekday procrastinator", "plays intramural soccer badly",
                      "collects vinyl I can't afford", "amateur baker, professional snacker"],
            "vibe": ["living on {food} and hope", "counting down to graduation",
                     "here for the memes and the free food", "just trying to pass exams",
                     "already counting days to {festival} break", "can't wait for {cricket}"],
        },
        emoji_pool=EMOJI_CASUAL, emoji_prob=0.5, parts_range=(2, 3),
        high_card_pool=DET_STUDENT,
    )


def bank_developer() -> C.PhraseBank:
    return C.PhraseBank(
        fragments={
            "role": ["Software engineer @ {company}", "Backend dev, {city}-based", "Full-stack developer",
                     "Building things at {company}", "Frontend dev who loves CSS more than I should",
                     "{first} — mobile engineer"],
            "stack": ["Python & Go by day", "React/TypeScript enjoyer", "Rust curious, Python fluent",
                      "into distributed systems", "cloud infra tinkerer", "occasional open-source contributor"],
            "hobby": ["debugging in production (don't tell my manager)", "side-project addict",
                      "mechanical keyboard collector", "coffee-powered", "board games on weekends"],
            "vibe": ["opinions are my own", "views != my employer's", "always shipping something small"],
        },
        emoji_pool=EMOJI_DEV, emoji_prob=0.4, parts_range=(2, 3),
        high_card_pool=DET_DEV,
    )


def bank_business(kind: str = "Account") -> C.PhraseBank:
    return C.PhraseBank(
        fragments={
            "role": [f"{{company}} — official {kind.lower()}", "Family-owned, {city}-based",
                     "Serving {city} since {year}", "Small business, big heart",
                     "Handmade goods from {city}", "Local favorite in {city}"],
            "offer": ["Custom orders welcome", "DM us for inquiries", "Free shipping over ₹999",
                      "New drops every {number} weeks", "Booking now for {year}",
                      "Quality you can trust", "Special {festival} offers running now"],
            "vibe": ["Proudly independent", "Woman-owned & operated", "Community first, always",
                     "Thank you for supporting small business", "Made in India, made with love"],
        },
        emoji_pool=EMOJI_BIZ, emoji_prob=0.3, parts_range=(2, 3),
        high_card_pool=DET_BIZ,
    )


def bank_fitness() -> C.PhraseBank:
    return C.PhraseBank(
        fragments={
            "role": ["Certified personal trainer", "Online coach @ {company}", "Powerlifter & coach",
                     "Helping you build strength, not just muscle", "{first} — movement is medicine",
                     "Marathoner turned strength coach"],
            "focus": ["macro-friendly recipes on the blog", "form check requests welcome",
                      "5am workouts, no excuses", "mobility nerd", "believer in progressive overload",
                      "recovering cardio-phobe turned lifter", "swapped fried snacks for home-cooked {food}"],
            "cta": ["Programs linked below", "DM 'START' for coaching info", "Free workout guide in bio",
                    "Booking 1:1 sessions"],
        },
        emoji_pool=EMOJI_FIT, emoji_prob=0.6, parts_range=(2, 3),
        high_card_pool=DET_FIT,
    )


def bank_spam() -> C.PhraseBank:
    return C.PhraseBank(
        fragments={
            "hook": ["MAKE $500/DAY FROM HOME", "Click the link and thank me later", "Follow for a shoutout!!",
                     "DM me 'INFO' now", "Everyone is doing this, why aren't you?", "Limited spots left!!"],
            "offer": ["Free crypto signals daily", "Guaranteed followers overnight", "Win a free iPhone today",
                      "Get verified fast, DM now", "Turn $10 into $1000", "Exclusive deal for my first 100 fans"],
            "cta": ["Link in bio NOW", "DM for details", "Tap the link before it's gone",
                    "Comment 'ME' below", "Check my story for proof"],
        },
        emoji_pool=EMOJI_SPAM, emoji_prob=0.8, parts_range=(2, 3),
        high_card_pool=DET_SPAM,
    )


def bank_fake_influencer() -> C.PhraseBank:
    return C.PhraseBank(
        fragments={
            "hook": ["Living my best life ✈️", "Full-time content creator", "Brand deals only, please",
                     "{first} | lifestyle & beauty", "Turning passion into a career", "Business inquiries only"],
            "offer": ["10k+ engaged followers, ask for my rate card", "Collab requests welcome",
                      "As seen on your explore page", "Partner with the biggest brands"],
            "cta": ["Email in bio for collabs", "Manager: link below", "Shop my looks — link in bio"],
        },
        emoji_pool=EMOJI_GLAM, emoji_prob=0.7, parts_range=(2, 3),
        high_card_pool=DET_INFLUENCER,
    )


def bank_celebrity_impersonator() -> C.PhraseBank:
    return C.PhraseBank(
        fragments={
            "hook": ["THIS IS MY ONLY REAL ACCOUNT", "Official page, all others are fake", "Not a bot, it's really me",
                     "My verified account got hacked, this is the new one", "Beware of fake accounts impersonating me"],
            "offer": ["Direct message me here for business", "I reply to my real fans personally",
                      "New project coming soon", "Thank you for {number} years of support"],
            "cta": ["DM me, I promise it's me", "Follow this account only", "Ignore the other fake pages"],
        },
        emoji_pool=EMOJI_CASUAL, emoji_prob=0.4, parts_range=(2, 2),
        high_card_pool=DET_IMPERSONATOR,
    )


def cap_generic(openers, bodies, ctas, emoji_pool, cta_prob=0.3, emoji_prob=0.5, details=None,
                 detail_prob=0.95, detail2_prob=0.0) -> C.CaptionBank:
    return C.CaptionBank(openers=openers, bodies=bodies, ctas=ctas, emoji_pool=emoji_pool,
                          cta_prob=cta_prob, emoji_prob=emoji_prob, details=details or [],
                          detail_prob=detail_prob, detail2_prob=detail2_prob)


CAP_STUDENT = cap_generic(
    openers=["Another day surviving {school}.", "Finals season has me questioning everything.",
             "Group project update:", "Campus is so pretty in {city} today.", "Coffee run before class.",
             "Study session with {first} turned into a nap."],
    bodies=["Somehow still passing my classes.", "My professor just dropped a pop quiz, send help.",
            "Library until it closes, again.", "Ranking my classes from bearable to soul-crushing.",
            "Turns out procrastination is a full-time job.", "Dorm life really tests your patience."],
    ctas=["Send snacks.", "Wish me luck.", "Tell me it gets easier.", "Someone quiz me please."],
    emoji_pool=EMOJI_CASUAL, details=DET_STUDENT,
)

CAP_DEVELOPER = cap_generic(
    openers=["Spent way too long on a bug today.", "Shipped a small feature at {company}.",
             "Refactoring old code I wrote a year ago.", "New side project idea just hit me at 2am.",
             "Finally got the tests passing.", "Learning a new part of the stack this week.",
             "Migrated a service to a new framework today.", "On-call was quiet for once.",
             "Rewrote a script that's been bugging me for weeks.", "Gave a lunch-and-learn at {company} today."],
    bodies=["Turned out to be a missing semicolon.", "The code review comments humbled me.",
            "Past me really did not comment anything.", "CI is green and I'm emotionally fine now.",
            "Docs were wrong but figured it out eventually.", "Pairing with a teammate made it so much faster.",
            "Turns out the cache was the whole problem.", "Linting caught it before it shipped, thankfully.",
            "The fix was smaller than the investigation.", "Naming things is still the hardest part."],
    ctas=["Repo link below.", "Curious how others solve this.", "Open to feedback.", "AMA about the stack."],
    emoji_pool=EMOJI_DEV, details=DET_DEV,
)

CAP_BUSINESS = cap_generic(
    openers=["New arrivals just dropped!", "Thank you {city} for another great week.",
             "Behind the scenes at the shop today.", "Restocked your favorites.",
             "Meet the team making it happen.", "Small batch, made with care."],
    bodies=["We appreciate every single one of you.", "Orders ship out within 2 business days.",
            "Handmade in small batches every week.", "Your support keeps our doors open.",
            "Custom requests are always welcome.", "Locally sourced, always."],
    ctas=["Shop the link in bio.", "DM us to order.", "Tag someone who'd love this.", "Limited stock available."],
    emoji_pool=EMOJI_BIZ, cta_prob=0.5, details=DET_BIZ,
)

CAP_FITNESS = cap_generic(
    openers=["Leg day recap.", "New PR today!", "Meal prep Sunday.", "Recovery day, mobility work only.",
             "Coaching client hit a big milestone.", "Early morning session before work.",
             "Back-to-back sessions today.", "Deload week starts now.",
             "Programming update for the week.", "Ran a form check clinic today."],
    bodies=["Form over ego, always.", "Consistency beats intensity most days.",
            "Progress isn't always linear, and that's fine.", "Fueling properly made a huge difference.",
            "Small wins add up over months.", "Rest days are still training days.",
            "Sleep is still the most underrated recovery tool.", "Slow and controlled beats fast and sloppy.",
            "Showing up matters more than the perfect plan.", "Strength carries over into everything else."],
    ctas=["Programs linked in bio.", "DM 'START' to work together.", "Save this for your next session.",
          "Tag your gym partner."],
    emoji_pool=EMOJI_FIT, cta_prob=0.5, details=DET_FIT,
)

CAP_SPAM = cap_generic(
    openers=["🚨 LAST CHANCE 🚨", "Everyone is asking how I did this so fast.", "Stop scrolling, read this.",
             "I wasn't going to share this but...", "This changed everything for me.", "You NEED to see this."],
    bodies=["Made more money this week than my old job paid monthly.", "It's 100% legit, I was skeptical too.",
            "Spots are filling up fast.", "No experience needed, just a phone.", "This is not a scam, I promise.",
            "Thousands already joined, don't miss out."],
    ctas=["Link in bio, click now.", "DM me 'START' immediately.", "Comment before it's gone.",
          "Tap the link in my story."],
    emoji_pool=EMOJI_SPAM, cta_prob=0.8, emoji_prob=0.8, details=DET_SPAM,
)

CAP_FAKE_INFLUENCER = cap_generic(
    openers=["Obsessed with this new find.", "Another day, another shoot.", "This brand sent me the best package.",
             "Golden hour never disappoints.", "Living for moments like this.", "Can't stop wearing this lately.",
             "New content coming your way soon.", "Today's shoot was a whole vibe.",
             "This might be my favorite collab yet.", "Unboxing my favorite package of the month."],
    bodies=["Use my code for a discount.", "So grateful for opportunities like this.",
            "This is exactly what my feed needed.", "Can't wait to show you more from this collab.",
            "My followers deserve the best recommendations.", "Partnering with brands I actually love.",
            "Honestly didn't expect to love this as much as I do.", "This is going straight into my everyday rotation.",
            "So many of you have been asking about this.", "Can't believe I get to call this my job."],
    ctas=["Link in bio for the discount.", "Code is my username at checkout.", "Shop this look now.",
          "DM for collab details."],
    emoji_pool=EMOJI_GLAM, cta_prob=0.6, details=DET_INFLUENCER, detail2_prob=0.5,
)

CAP_CELEBRITY_IMPERSONATOR = cap_generic(
    openers=["To my real fans:", "Thank you for the love this {year}.", "Important announcement coming soon.",
             "I don't get to say this enough.", "This account is really me, I promise.",
             "Taking a moment to say thank you.", "Something special is coming, stay tuned.",
             "A quick note from me to you.", "I see all of your messages, truly."],
    bodies=["Please report any fake pages you see.", "I read every message even if I can't reply to all.",
            "New project details coming very soon.", "Grateful for every one of you every single day.",
            "Ignore anyone claiming to be my manager elsewhere.", "None of this would matter without you.",
            "This journey keeps surprising me.", "There's more coming that I can't wait to share."],
    ctas=["DM me directly here.", "Follow only this page.", "Share this so others know it's real."],
    emoji_pool=EMOJI_CASUAL, cta_prob=0.4, details=DET_IMPERSONATOR, detail2_prob=0.5,
)


# ===========================================================================
# Platform builders
# ===========================================================================

def instagram_archetypes() -> list[Archetype]:
    return [
        Archetype("Student", 0, bank_student(), CAP_STUDENT, HASHTAGS_STUDENT, username_style="casual",
                  hashtag_range=(1, 4), hashtag_empty_prob=0.1, account_age_range=(300, 3200),
                  primary_range=(50, 3000), secondary_mode="independent", secondary_range=(80, 1200),
                  engagement_rate_range=(0.03, 0.09), comment_ratio_range=(0.06, 0.18),
                  share_ratio_range=(0.01, 0.04), post_noise_sigma_range=(0.35, 0.6),
                  is_verified_p=0.005, has_website_p=0.05, has_location_p=0.35,
                  profile_completion_range=(0.45, 0.85)),
        Archetype("Developer", 0, bank_developer(), CAP_DEVELOPER, HASHTAGS_DEV, username_style="casual",
                  hashtag_range=(1, 3), hashtag_empty_prob=0.15, account_age_range=(300, 3500),
                  primary_range=(100, 8000), secondary_mode="independent", secondary_range=(50, 900),
                  engagement_rate_range=(0.02, 0.07), comment_ratio_range=(0.05, 0.15),
                  share_ratio_range=(0.01, 0.03), post_noise_sigma_range=(0.35, 0.6),
                  is_verified_p=0.01, has_website_p=0.35, has_location_p=0.3,
                  profile_completion_range=(0.5, 0.9)),
        Archetype("Business Account", 0, bank_business("Account"), CAP_BUSINESS, HASHTAGS_BIZ,
                  username_style="corporate", hashtag_range=(2, 5), hashtag_empty_prob=0.05,
                  account_age_range=(300, 4000), primary_range=(300, 60000),
                  secondary_mode="independent", secondary_range=(50, 2000),
                  engagement_rate_range=(0.015, 0.05), comment_ratio_range=(0.04, 0.12),
                  share_ratio_range=(0.01, 0.05), post_noise_sigma_range=(0.3, 0.55),
                  is_verified_p=0.03, has_website_p=0.7, has_location_p=0.8,
                  profile_completion_range=(0.6, 0.95)),
        Archetype("Fitness Creator", 0, bank_fitness(), CAP_FITNESS, HASHTAGS_FIT,
                  username_style="aesthetic", hashtag_range=(2, 6), hashtag_empty_prob=0.03,
                  account_age_range=(250, 3600), primary_range=(800, 250000),
                  secondary_mode="independent", secondary_range=(200, 3500),
                  engagement_rate_range=(0.03, 0.1), comment_ratio_range=(0.05, 0.15),
                  share_ratio_range=(0.02, 0.06), post_noise_sigma_range=(0.35, 0.6),
                  is_verified_p=0.06, has_website_p=0.55, has_location_p=0.5,
                  profile_completion_range=(0.55, 0.95)),
        Archetype("Spam Account", 1, bank_spam(), CAP_SPAM, HASHTAGS_SPAM, username_style="spammy",
                  hashtag_range=(3, 8), hashtag_empty_prob=0.02, account_age_range=(1, 365),
                  primary_range=(20, 4000), secondary_mode="ratio_near_1", ratio_range=(0.8, 1.4),
                  engagement_rate_range=(0.001, 0.01), comment_ratio_range=(0.01, 0.05),
                  share_ratio_range=(0.0, 0.02), post_noise_sigma_range=(0.5, 0.9),
                  is_verified_p=0.0, has_website_p=0.4, has_location_p=0.05,
                  profile_completion_range=(0.1, 0.4)),
        Archetype("Fake Influencer", 1, bank_fake_influencer(), CAP_FAKE_INFLUENCER, HASHTAGS_INFLUENCER,
                  username_style="aesthetic", hashtag_range=(2, 6), hashtag_empty_prob=0.05,
                  account_age_range=(30, 900), primary_range=(5000, 500000),
                  secondary_mode="independent", secondary_range=(50, 800),
                  engagement_rate_range=(0.001, 0.008), comment_ratio_range=(0.02, 0.08),
                  share_ratio_range=(0.0, 0.02), post_noise_sigma_range=(0.5, 0.85),
                  is_verified_p=0.02, has_website_p=0.6, has_location_p=0.2,
                  profile_completion_range=(0.4, 0.8)),
        Archetype("Celebrity Impersonator", 1, bank_celebrity_impersonator(), CAP_CELEBRITY_IMPERSONATOR,
                  HASHTAGS_CELEB, username_style="impersonator", hashtag_range=(0, 3), hashtag_empty_prob=0.3,
                  account_age_range=(1, 500), primary_range=(500, 100000),
                  secondary_mode="ratio_near_1", ratio_range=(0.7, 1.5),
                  engagement_rate_range=(0.002, 0.02), comment_ratio_range=(0.03, 0.1),
                  share_ratio_range=(0.0, 0.03), post_noise_sigma_range=(0.5, 0.9),
                  is_verified_p=0.0, has_website_p=0.1, has_location_p=0.1,
                  profile_completion_range=(0.5, 0.9)),
    ]


def twitter_archetypes() -> list[Archetype]:
    caps = {
        "Student": CAP_STUDENT, "Developer": CAP_DEVELOPER, "Business Account": CAP_BUSINESS,
        "Fitness Creator": CAP_FITNESS, "Spam/Bot Account": CAP_SPAM,
        "Fake Influencer": CAP_FAKE_INFLUENCER, "Celebrity Impersonator": CAP_CELEBRITY_IMPERSONATOR,
    }
    base = instagram_archetypes()
    out = []
    for a in base:
        name = "Spam/Bot Account" if a.name == "Spam Account" else a.name
        out.append(Archetype(
            name=name, is_fake=a.is_fake, text_bank=a.text_bank, caption_bank=caps[name],
            hashtag_pool=a.hashtag_pool, username_style=a.username_style,
            hashtag_range=(max(0, a.hashtag_range[0] - 1), max(1, a.hashtag_range[1] - 1)),
            hashtag_empty_prob=min(0.5, a.hashtag_empty_prob + 0.15),
            account_age_range=a.account_age_range, primary_range=a.primary_range,
            secondary_mode=a.secondary_mode, secondary_range=a.secondary_range, ratio_range=a.ratio_range,
            engagement_rate_range=a.engagement_rate_range, comment_ratio_range=a.comment_ratio_range,
            share_ratio_range=a.share_ratio_range, post_noise_sigma_range=a.post_noise_sigma_range,
            is_verified_p=a.is_verified_p, has_website_p=a.has_website_p, has_location_p=a.has_location_p,
            profile_completion_range=a.profile_completion_range,
        ))
    return out


def bank_catfish() -> C.PhraseBank:
    return C.PhraseBank(
        fragments={
            "hook": ["Widowed, looking for genuine connection", "Military stationed overseas, hard to talk on phone",
                      "New here, hoping to meet someone real", "Engineer working abroad on an oil rig",
                      "Single parent, honest and loyal", "Doctor currently on a mission overseas"],
            "trait": ["Believer in true love", "Not into games, just want honesty", "Family means everything to me",
                       "Looking for my person", "Tired of being lied to, hoping you're different"],
            "cta": ["Message me, let's talk", "Add me, I don't bite", "Let's chat somewhere more private"],
        },
        emoji_pool=["❤️", "🙏", "🌹", "💌"], emoji_prob=0.5, parts_range=(2, 2),
        high_card_pool=DET_CATFISH,
    )


CAP_CATFISH = cap_generic(
    openers=["Thinking about you today.", "Another lonely night here overseas.", "Can't wait to finally meet.",
             "Just need someone to talk to.", "Missing having someone real in my life.",
             "Long day, but talking to you helps.", "Wish you were here with me right now.",
             "Counting down the days already."],
    bodies=["Work keeps me so busy but you're always on my mind.", "I don't trust easily but you're different.",
            "Money's been tight since the last assignment.", "Hoping we can talk more privately soon.",
            "I promise I'm not like the others you've met online.", "You've made this whole assignment easier.",
            "Every message from you makes my day better."],
    ctas=["Message me on here.", "Let's move the conversation elsewhere.", "Please don't give up on us."],
    emoji_pool=["❤️", "🌹", "🙏"], cta_prob=0.4, details=DET_CATFISH, detail2_prob=0.5,
)


def facebook_archetypes() -> list[Archetype]:
    return [
        Archetype("Student", 0, bank_student(), CAP_STUDENT, HASHTAGS_STUDENT, username_style="casual",
                  hashtag_range=(0, 2), hashtag_empty_prob=0.5, account_age_range=(300, 3200),
                  primary_range=(80, 1500), secondary_mode="independent", secondary_range=(0, 30),
                  engagement_rate_range=(0.05, 0.15), comment_ratio_range=(0.1, 0.3),
                  share_ratio_range=(0.02, 0.08), post_noise_sigma_range=(0.35, 0.6),
                  is_verified_p=0.002, has_website_p=0.03, has_location_p=0.5,
                  profile_completion_range=(0.5, 0.85)),
        Archetype("Developer", 0, bank_developer(), CAP_DEVELOPER, HASHTAGS_DEV, username_style="casual",
                  hashtag_range=(0, 2), hashtag_empty_prob=0.6, account_age_range=(300, 3500),
                  primary_range=(100, 2000), secondary_mode="independent", secondary_range=(0, 40),
                  engagement_rate_range=(0.03, 0.1), comment_ratio_range=(0.08, 0.2),
                  share_ratio_range=(0.01, 0.05), post_noise_sigma_range=(0.35, 0.6),
                  is_verified_p=0.005, has_website_p=0.3, has_location_p=0.4,
                  profile_completion_range=(0.5, 0.9)),
        Archetype("Business Page", 0, bank_business("Page"), CAP_BUSINESS, HASHTAGS_BIZ,
                  username_style="corporate", hashtag_range=(0, 3), hashtag_empty_prob=0.35,
                  account_age_range=(300, 4000), primary_range=(0, 60), secondary_mode="independent",
                  secondary_range=(300, 50000),
                  engagement_rate_range=(0.01, 0.05), comment_ratio_range=(0.05, 0.15),
                  share_ratio_range=(0.02, 0.08), post_noise_sigma_range=(0.3, 0.55),
                  is_verified_p=0.04, has_website_p=0.75, has_location_p=0.85,
                  profile_completion_range=(0.6, 0.95)),
        Archetype("Fitness Creator", 0, bank_fitness(), CAP_FITNESS, HASHTAGS_FIT,
                  username_style="aesthetic", hashtag_range=(0, 3), hashtag_empty_prob=0.3,
                  account_age_range=(250, 3600), primary_range=(100, 3000), secondary_mode="independent",
                  secondary_range=(500, 150000),
                  engagement_rate_range=(0.02, 0.08), comment_ratio_range=(0.06, 0.18),
                  share_ratio_range=(0.02, 0.07), post_noise_sigma_range=(0.35, 0.6),
                  is_verified_p=0.05, has_website_p=0.5, has_location_p=0.5,
                  profile_completion_range=(0.55, 0.95)),
        Archetype("Spam Account", 1, bank_spam(), CAP_SPAM, HASHTAGS_SPAM, username_style="spammy",
                  hashtag_range=(0, 3), hashtag_empty_prob=0.4, account_age_range=(1, 365),
                  primary_range=(10, 800), secondary_mode="ratio_near_1", ratio_range=(0.7, 1.3),
                  engagement_rate_range=(0.001, 0.01), comment_ratio_range=(0.01, 0.05),
                  share_ratio_range=(0.0, 0.02), post_noise_sigma_range=(0.5, 0.9),
                  is_verified_p=0.0, has_website_p=0.35, has_location_p=0.05,
                  profile_completion_range=(0.1, 0.4)),
        Archetype("Fake Influencer", 1, bank_fake_influencer(), CAP_FAKE_INFLUENCER, HASHTAGS_INFLUENCER,
                  username_style="aesthetic", hashtag_range=(0, 3), hashtag_empty_prob=0.4,
                  account_age_range=(30, 900), primary_range=(50, 700), secondary_mode="independent",
                  secondary_range=(3000, 300000),
                  engagement_rate_range=(0.001, 0.008), comment_ratio_range=(0.02, 0.08),
                  share_ratio_range=(0.0, 0.02), post_noise_sigma_range=(0.5, 0.85),
                  is_verified_p=0.02, has_website_p=0.55, has_location_p=0.15,
                  profile_completion_range=(0.4, 0.8)),
        Archetype("Celebrity Impersonator", 1, bank_celebrity_impersonator(), CAP_CELEBRITY_IMPERSONATOR,
                  HASHTAGS_CELEB, username_style="impersonator", hashtag_range=(0, 2), hashtag_empty_prob=0.5,
                  account_age_range=(1, 500), primary_range=(20, 600), secondary_mode="independent",
                  secondary_range=(500, 80000),
                  engagement_rate_range=(0.002, 0.02), comment_ratio_range=(0.03, 0.1),
                  share_ratio_range=(0.0, 0.03), post_noise_sigma_range=(0.5, 0.9),
                  is_verified_p=0.0, has_website_p=0.1, has_location_p=0.1,
                  profile_completion_range=(0.5, 0.9)),
        Archetype("Catfish/Romance-Scam Profile", 1, bank_catfish(), CAP_CATFISH, [],
                  username_style="casual", hashtag_range=(0, 0), hashtag_empty_prob=1.0,
                  account_age_range=(1, 200), primary_range=(5, 150), secondary_mode="independent",
                  secondary_range=(0, 10),
                  engagement_rate_range=(0.02, 0.08), comment_ratio_range=(0.1, 0.3),
                  share_ratio_range=(0.0, 0.02), post_noise_sigma_range=(0.5, 0.8),
                  is_verified_p=0.0, has_website_p=0.02, has_location_p=0.05,
                  profile_completion_range=(0.2, 0.55)),
    ]


def linkedin_archetypes() -> list[Archetype]:
    cap_student_li = cap_generic(
        openers=["Excited to share I just finished a project for class.", "Attended a great career fair today.",
                 "Grateful for an amazing internship experience this summer.", "Just wrapped up finals!",
                 "Reflecting on my first year at {school}.", "Presented my capstone project today.",
                 "Joined a new student organization this semester.", "Had a great conversation with a recruiter today.",
                 "Just finished my first group case competition."],
        bodies=["Learned so much about teamwork and deadlines.", "Met some incredible recruiters and alumni.",
                "This confirmed the career path I want to pursue.", "Time to recharge before next semester.",
                "Grateful for professors who go above and beyond.", "Still processing everything I learned this week.",
                "Balancing coursework and applications has been a lot, but worth it.",
                "This experience changed how I think about my major."],
        ctas=["Open to internship opportunities.", "Would love to connect with others in the field.",
              "Feel free to reach out!", "Always happy to compare notes with fellow students."],
        emoji_pool=[], cta_prob=0.4, emoji_prob=0.05, details=DET_LI_STUDENT, detail2_prob=0.7,
    )
    cap_dev_li = cap_generic(
        openers=["Shipped a new feature at {company} this week.", "Wrote a short post on lessons from a recent outage.",
                 "Excited to start using a new framework on our team.", "Reflecting on 3 years as a developer.",
                 "Mentored a junior engineer today and loved it.", "Migrated a legacy service this week.",
                 "Gave a short talk on our team's testing practices.", "Finally closed out a long-standing tech debt ticket.",
                 "Onboarded a new teammate this week."],
        bodies=["Documentation really does save future you.", "Cross-team collaboration made this so much smoother.",
                "Testing early saved us a painful rollback.", "Grateful for a team that values code quality.",
                "Always learning something new in this role.", "The postmortem taught us more than the incident did.",
                "Small, frequent deploys keep saving us from bigger headaches.",
                "Good tooling makes the whole team faster."],
        ctas=["Happy to share more details if useful.", "Open to connecting with other engineers.",
              "Let me know your thoughts below.", "Always glad to swap notes with other teams."],
        emoji_pool=[], cta_prob=0.3, emoji_prob=0.05, details=DET_LI_DEV, detail2_prob=0.7,
    )
    cap_biz_li = cap_generic(
        openers=["Proud to announce a new partnership.", "Our team just hit a major milestone.",
                 "Thank you to our clients for another great quarter.", "Excited to share our latest case study.",
                 "We're hiring! Come join our growing team.", "Reflecting on another strong quarter for the team.",
                 "Welcoming several new team members this month.", "Excited to unveil our latest product update.",
                 "Grateful for the recognition from our industry peers."],
        bodies=["This wouldn't be possible without our incredible team.", "Looking forward to what's next for us.",
                "Grateful for the trust our clients place in us.", "Innovation continues to drive everything we do.",
                "We're expanding into new markets this year.", "Every milestone here is a team effort.",
                "Our clients' feedback keeps shaping the roadmap.", "Culture and craft both matter to how we build."],
        ctas=["Learn more on our website.", "Reach out if you'd like to collaborate.",
              "Apply via the link in our profile.", "We'd love to hear from you."],
        emoji_pool=[], cta_prob=0.5, emoji_prob=0.02, details=DET_LI_BIZ, detail2_prob=0.7,
    )
    cap_consultant_li = cap_generic(
        openers=["Wrapped up an engagement with a great client this week.", "Some thoughts on change management.",
                 "Just published a short case study.", "Speaking at a panel next month on industry trends.",
                 "Reflecting on 10 years of consulting.", "Kicked off a new engagement this week.",
                 "Facilitated a strategy workshop today.", "Wrapped up a multi-month transformation project.",
                 "Sharing a few lessons from a recent client engagement."],
        bodies=["Strategy is only as good as its execution.", "Every client engagement teaches me something new.",
                "The best solutions come from listening first.", "Grateful to work across such varied industries.",
                "Frameworks help, but context always wins.", "Alignment across stakeholders matters more than the plan itself.",
                "The hardest part is rarely the analysis, it's the change management.",
                "Every industry has more in common than people expect."],
        ctas=["Happy to discuss further, DM me.", "Would love your thoughts in the comments.",
              "Reach out if this resonates.", "Open to new client conversations."],
        emoji_pool=[], cta_prob=0.35, emoji_prob=0.02, details=DET_LI_CONSULTANT, detail2_prob=0.7,
    )
    cap_fake_recruiter = cap_generic(
        openers=["URGENT HIRING: multiple positions open now!", "We are looking for talent, apply immediately!",
                 "Exciting remote opportunity, no experience needed!", "Hiring managers are reviewing applications TODAY.",
                 "Dream job alert, don't wait on this one!", "Companies are hiring fast this quarter, apply now!",
                 "This role won't stay open long!"],
        bodies=["High salary, flexible hours guaranteed.", "Send your resume and bank details to get started fast.",
                "This opportunity won't last, act now.", "Hundreds already applied, don't miss out.",
                "No interview required for qualified candidates.", "We place candidates faster than anyone else.",
                "Positions are filling up quickly this week."],
        ctas=["DM me your resume today.", "Apply via the link in my profile now.", "Message me ASAP to secure your spot.",
              "Comment 'HIRE ME' below."],
        emoji_pool=["🚀", "💰", "✅"], cta_prob=0.7, emoji_prob=0.5, details=DET_LI_RECRUITER, detail2_prob=0.7,
    )
    cap_fake_exec = cap_generic(
        openers=["As CEO, I'm proud to announce a groundbreaking initiative.", "Leading the industry into a new era.",
                 "My latest leadership insight for aspiring executives.", "Announcing a major company milestone under my leadership.",
                 "Another record quarter under my leadership.", "Here's the leadership lesson nobody tells you.",
                 "Sharing my vision for where this industry is headed."],
        bodies=["True leadership means thinking bigger than everyone else.", "We're disrupting the entire sector.",
                "My journey to the top wasn't easy, but it was inevitable.", "Success comes to those who never settle.",
                "Most executives think too small, I never have.", "Great leaders create their own opportunities.",
                "This is only the beginning of what we're building."],
        ctas=["Connect with me for exclusive opportunities.", "Follow for more leadership wisdom.",
              "DM me to discuss investment opportunities.", "Reach out if you want to learn more."],
        emoji_pool=["💼", "🚀"], cta_prob=0.5, emoji_prob=0.2, details=DET_LI_EXEC, detail2_prob=0.7,
    )
    cap_spam_li = cap_generic(
        openers=["Grow your network 10x with this simple trick.", "I made $10k this month using this LinkedIn hack.",
                 "Everyone should be doing this in 2026.", "Stop scrolling, this could change your career.",
                 "The algorithm doesn't want you to know this.", "This one trick changed my whole career trajectory.",
                 "I wish someone told me this five years ago."],
        bodies=["DM me and I'll show you exactly how.", "This strategy works for anyone, guaranteed.",
                "Comment 'INFO' and I'll send details.", "Limited spots in my mentorship program.",
                "I've helped hundreds of people do the same.", "It's simpler than you'd expect, promise.",
                "This isn't some overnight gimmick, it actually works."],
        ctas=["Click the link in my profile.", "DM me the word 'START'.", "Book a free call today.",
              "Comment below and I'll follow up."],
        emoji_pool=["🚀", "📈", "💯"], cta_prob=0.7, emoji_prob=0.5, details=DET_LI_SPAM, detail2_prob=0.7,
    )

    li_hash_generic = ["career", "networking", "hiring", "leadership", "innovation"]
    li_hash_spam = ["hiring", "remotework", "makemoney", "opportunity", "grind"]

    def bank_li_student():
        return C.PhraseBank(
            fragments={
                "role": ["Student at {school}", "{first} | undergraduate student", "Graduate student at {school}",
                          "Aspiring professional, studying at {school}", "Undergrad at {school}, class of {year}",
                          "{first} | student researcher at {school}", "Studying in {city}, always curious"],
                "focus": ["passionate about learning and growth", "seeking internship opportunities",
                          "active in student organizations", "combining coursework with real-world projects",
                          "eager to apply classroom learning to real problems", "building a portfolio one project at a time",
                          "always looking for the next challenge"],
            },
            emoji_pool=[], emoji_prob=0.05, parts_range=(2, 2), high_card_pool=DET_LI_STUDENT,
        )

    def bank_li_dev():
        return C.PhraseBank(
            fragments={
                "role": ["Software Engineer at {company}", "Backend Developer | {city}",
                          "Full-Stack Engineer building scalable systems", "Engineer @ {company}",
                          "{first} | Software Engineer", "Platform Engineer based in {city}",
                          "Mobile Engineer @ {company}"],
                "focus": ["passionate about clean code and mentorship", "focused on distributed systems",
                          "enjoys solving hard technical problems", "advocate for good engineering practices",
                          "believer in small, well-tested changes", "always tinkering with a side project",
                          "focused on developer experience and tooling"],
            },
            emoji_pool=[], emoji_prob=0.05, parts_range=(2, 2), high_card_pool=DET_LI_DEV,
        )

    def bank_li_biz():
        return C.PhraseBank(
            fragments={
                "role": ["{company} | Official Page", "Helping businesses grow since {year}",
                          "{city}-based company delivering results", "Innovating in our industry since {year}",
                          "{company}, headquartered in {city}", "Serving clients worldwide since {year}",
                          "Built in {city}, trusted everywhere"],
                "focus": ["dedicated to client success", "committed to quality and innovation",
                          "proud to serve customers worldwide", "building a great place to work",
                          "focused on long-term partnerships", "driven by our customers' success",
                          "investing in our people and our product"],
            },
            emoji_pool=[], emoji_prob=0.02, parts_range=(2, 2), high_card_pool=DET_LI_BIZ,
        )

    def bank_li_consultant():
        return C.PhraseBank(
            fragments={
                "role": ["Independent Consultant | Strategy", "Management Consultant @ {company}",
                          "Helping companies navigate change", "{first} | Consultant & advisor",
                          "Consultant based in {city}", "Advisor to leadership teams since {year}",
                          "Strategy Consultant @ {company}"],
                "focus": ["specializing in operations and growth", "20+ engagements across industries",
                          "focused on measurable outcomes", "trusted advisor to leadership teams",
                          "helping teams turn strategy into execution", "bringing outside perspective to hard problems",
                          "focused on practical, durable change"],
            },
            emoji_pool=[], emoji_prob=0.02, parts_range=(2, 2), high_card_pool=DET_LI_CONSULTANT,
        )

    def bank_li_fake_recruiter():
        return C.PhraseBank(
            fragments={
                "role": ["Senior Talent Acquisition Specialist", "Global Recruiter | Hiring Now",
                          "Connecting talent with opportunity worldwide", "Recruiter @ {company}",
                          "Talent Partner based in {city}", "Recruiting for {company} and partners",
                          "Hiring Manager | {city} and remote"],
                "focus": ["hiring for multiple remote roles", "helping candidates land dream jobs fast",
                          "always looking for new talent", "high-paying opportunities available now",
                          "placing candidates across every industry", "filling roles faster than anyone else",
                          "connecting top talent with top pay"],
            },
            emoji_pool=["🚀"], emoji_prob=0.3, parts_range=(2, 2), high_card_pool=DET_LI_RECRUITER,
        )

    def bank_li_fake_exec():
        return C.PhraseBank(
            fragments={
                "role": ["CEO & Founder | Visionary Leader", "Chairman of {company}", "Serial Entrepreneur | CEO",
                          "President & CEO, disrupting the industry", "Founder & CEO @ {company}",
                          "CEO | Based in {city}, thinking globally", "Chairman & Founder since {year}"],
                "focus": ["leading global teams to success", "building the next big thing",
                          "featured thought leader in business", "top 1% of executives worldwide",
                          "scaling companies from zero to global", "obsessed with outsized growth",
                          "redefining what's possible in this industry"],
            },
            emoji_pool=["💼"], emoji_prob=0.3, parts_range=(2, 2), high_card_pool=DET_LI_EXEC,
        )

    def bank_li_spam():
        return C.PhraseBank(
            fragments={
                "role": ["Growth Hacker | LinkedIn Top Voice", "Helping you 10x your network",
                          "Digital Marketing Guru", "Passive Income Coach", "Online Business Mentor @ {company}",
                          "Financial Freedom Coach | {city}", "Founder of my own success system"],
                "focus": ["teaching thousands how to succeed online", "DM me to learn my system",
                          "self-made success story", "helping others achieve financial freedom",
                          "turning strangers into six-figure success stories", "sharing the system that changed my life",
                          "building a community of driven entrepreneurs"],
            },
            emoji_pool=["🚀", "💰"], emoji_prob=0.4, parts_range=(2, 2), high_card_pool=DET_LI_SPAM,
        )

    return [
        Archetype("Student", 0, bank_li_student(), cap_student_li, li_hash_generic, username_style="casual",
                  hashtag_range=(0, 2), hashtag_empty_prob=0.6, account_age_range=(200, 2500),
                  primary_range=(50, 800), secondary_mode="none",
                  engagement_rate_range=(0.02, 0.08), comment_ratio_range=(0.1, 0.3),
                  share_ratio_range=(0.02, 0.08), post_noise_sigma_range=(0.35, 0.6),
                  is_verified_p=0.001, has_website_p=0.05, has_location_p=0.5,
                  profile_completion_range=(0.4, 0.8)),
        Archetype("Developer", 0, bank_li_dev(), cap_dev_li, li_hash_generic, username_style="casual",
                  hashtag_range=(0, 2), hashtag_empty_prob=0.6, account_age_range=(300, 3500),
                  primary_range=(200, 5000), secondary_mode="none",
                  engagement_rate_range=(0.015, 0.06), comment_ratio_range=(0.08, 0.2),
                  share_ratio_range=(0.02, 0.06), post_noise_sigma_range=(0.35, 0.6),
                  is_verified_p=0.005, has_website_p=0.3, has_location_p=0.5,
                  profile_completion_range=(0.5, 0.9)),
        Archetype("Business/Company Page", 0, bank_li_biz(), cap_biz_li, li_hash_generic,
                  username_style="corporate", hashtag_range=(0, 3), hashtag_empty_prob=0.45,
                  account_age_range=(400, 4000), primary_range=(500, 30000), secondary_mode="none",
                  engagement_rate_range=(0.01, 0.05), comment_ratio_range=(0.05, 0.15),
                  share_ratio_range=(0.02, 0.08), post_noise_sigma_range=(0.3, 0.55),
                  is_verified_p=0.05, has_website_p=0.85, has_location_p=0.9,
                  profile_completion_range=(0.65, 0.97)),
        Archetype("Consultant", 0, bank_li_consultant(), cap_consultant_li, li_hash_generic,
                  username_style="casual", hashtag_range=(0, 3), hashtag_empty_prob=0.5,
                  account_age_range=(400, 4000), primary_range=(500, 15000), secondary_mode="none",
                  engagement_rate_range=(0.015, 0.06), comment_ratio_range=(0.08, 0.2),
                  share_ratio_range=(0.02, 0.07), post_noise_sigma_range=(0.35, 0.6),
                  is_verified_p=0.01, has_website_p=0.4, has_location_p=0.6,
                  profile_completion_range=(0.55, 0.92)),
        Archetype("Fake Recruiter", 1, bank_li_fake_recruiter(), cap_fake_recruiter, li_hash_spam,
                  username_style="spammy", hashtag_range=(1, 3), hashtag_empty_prob=0.2,
                  account_age_range=(1, 400), primary_range=(1000, 29000), secondary_mode="none",
                  engagement_rate_range=(0.001, 0.01), comment_ratio_range=(0.02, 0.08),
                  share_ratio_range=(0.0, 0.02), post_noise_sigma_range=(0.5, 0.9),
                  is_verified_p=0.0, has_website_p=0.2, has_location_p=0.1,
                  profile_completion_range=(0.2, 0.5)),
        Archetype("Fake Executive Impersonator", 1, bank_li_fake_exec(), cap_fake_exec, li_hash_generic,
                  username_style="impersonator", hashtag_range=(0, 2), hashtag_empty_prob=0.4,
                  account_age_range=(1, 300), primary_range=(50, 3000), secondary_mode="none",
                  engagement_rate_range=(0.002, 0.015), comment_ratio_range=(0.03, 0.1),
                  share_ratio_range=(0.0, 0.03), post_noise_sigma_range=(0.5, 0.9),
                  is_verified_p=0.0, has_website_p=0.15, has_location_p=0.15,
                  profile_completion_range=(0.35, 0.75)),
        Archetype("Spam/Bot Account", 1, bank_li_spam(), cap_spam_li, li_hash_spam,
                  username_style="spammy", hashtag_range=(1, 4), hashtag_empty_prob=0.15,
                  account_age_range=(1, 365), primary_range=(500, 29000), secondary_mode="none",
                  engagement_rate_range=(0.001, 0.008), comment_ratio_range=(0.01, 0.05),
                  share_ratio_range=(0.0, 0.02), post_noise_sigma_range=(0.5, 0.9),
                  is_verified_p=0.0, has_website_p=0.3, has_location_p=0.05,
                  profile_completion_range=(0.1, 0.4)),
    ]


# ---------------------------------------------------------------------------
# LinkedIn-specific extra fields: job_title, company_name, endorsements, skills
# ---------------------------------------------------------------------------

JOB_TITLES_GENUINE = {
    "Student": ["Undergraduate Student", "Graduate Student", "Teaching Assistant", "Research Assistant"],
    "Developer": ["Software Engineer", "Backend Developer", "Frontend Developer", "Full-Stack Engineer",
                  "DevOps Engineer", "Mobile Engineer"],
    "Business/Company Page": ["Company Page", "Corporate Account"],
    "Consultant": ["Management Consultant", "Independent Consultant", "Strategy Consultant", "Business Advisor"],
}
JOB_TITLES_FAKE_RECRUITER = ["Talent Acquisition Specialist", "Global Recruiter", "HR Manager & Recruiter",
                             "Senior Recruitment Consultant", "Head of Global Hiring"]
JOB_TITLES_FAKE_EXEC = ["CEO & Founder", "Chairman & CEO", "President & Chief Executive Officer",
                         "Founder & Managing Director", "Executive Chairman"]
JOB_TITLES_SPAM = ["Growth Hacker", "Digital Marketing Guru", "Passive Income Coach", "LinkedIn Top Voice",
                    "Online Business Mentor"]

IMPLAUSIBLE_COMPANIES = ["Global Synergy Holdings", "Apex World Enterprises", "Nexus International Group",
                          "Prime Vantage Corp", "Meridian Worldwide Inc", "Zenith Capital Partners",
                          "Vertex Global Solutions", "Pinnacle Universal Holdings"]


LI_GENUINE_ENDORSEMENT_OVERLAP = (0, 10)     # what a fake account's endorsements typically look like
LI_GENUINE_SKILLS_OVERLAP = (0, 5)
LI_FAKE_ENDORSEMENT_OVERLAP = (60, 350)      # what a genuine account's endorsements typically look like
LI_FAKE_SKILLS_OVERLAP = (8, 45)
LI_GENUINE_RECOMMENDATION_OVERLAP = (0, 1)   # what a fake account's recommendation count typically looks like
LI_FAKE_RECOMMENDATION_OVERLAP = (3, 12)     # what a genuine account's recommendation count typically looks like
LI_GENUINE_ACCEPTANCE_OVERLAP = (0.05, 0.3)  # what a fake account's connection acceptance rate typically looks like
LI_FAKE_ACCEPTANCE_OVERLAP = (0.35, 0.85)    # what a genuine account's connection acceptance rate typically looks like


def _apply_overlap(rng, arr, n, own_is_fake, genuine_overlap_range, fake_overlap_range, sampler):
    """genuine_overlap_range/fake_overlap_range each hold the value INJECTED INTO that
    class (genuine_overlap_range is fake-typical content applied to genuine rows, and
    vice versa) -- see the matching comment in build_platform_dataset."""
    n_ov = int(round(n * OVERLAP_FRAC))
    if n_ov <= 0:
        return arr
    idx = rng.choice(n, size=n_ov, replace=False)
    opp_range = genuine_overlap_range if own_is_fake == 0 else fake_overlap_range
    arr[idx] = sampler(rng, *opp_range, n_ov)
    return arr


def _indian_company(rng) -> str:
    return C.INDIAN_COMPANY_NAMES[int(rng.integers(0, len(C.INDIAN_COMPANY_NAMES)))]


def linkedin_extra_builders() -> dict[str, ExtraFieldBuilder]:
    def genuine_builder(archetype_name: str) -> ExtraFieldBuilder:
        titles = JOB_TITLES_GENUINE[archetype_name]

        def build(rng, fk, n):
            job_title = [titles[int(rng.integers(0, len(titles)))] for _ in range(n)]
            company_name = [_indian_company(rng) for _ in range(n)]
            endorsements = np.round(C.sample_lognormal_clipped(rng, 0, 400, n)).astype(int)
            skills = np.round(C.sample_uniform_float(rng, 3, 50, n)).astype(int)
            endorsements = _apply_overlap(rng, endorsements, n, 0, LI_GENUINE_ENDORSEMENT_OVERLAP,
                                           LI_FAKE_ENDORSEMENT_OVERLAP,
                                           lambda r, lo, hi, k: np.round(C.sample_lognormal_clipped(r, lo, hi, k)).astype(int))
            skills = _apply_overlap(rng, skills, n, 0, LI_GENUINE_SKILLS_OVERLAP, LI_FAKE_SKILLS_OVERLAP,
                                     lambda r, lo, hi, k: np.round(C.sample_uniform_float(r, lo, hi, k)).astype(int))
            # Written recommendations are far harder to fake at scale than one-click
            # endorsements (they require a real second party to write real text), so
            # genuine profiles carry a modest but real count; connection acceptance
            # sits in the normal professional-networking range.
            recommendations = np.round(C.sample_lognormal_clipped(rng, 0, 12, n)).astype(int)
            acceptance = np.round(C.sample_uniform_float(rng, 0.35, 0.85, n), 3)
            recommendations = _apply_overlap(rng, recommendations, n, 0, LI_GENUINE_RECOMMENDATION_OVERLAP,
                                              LI_FAKE_RECOMMENDATION_OVERLAP,
                                              lambda r, lo, hi, k: np.round(C.sample_lognormal_clipped(r, lo, hi, k)).astype(int))
            acceptance = _apply_overlap(rng, acceptance, n, 0, LI_GENUINE_ACCEPTANCE_OVERLAP,
                                         LI_FAKE_ACCEPTANCE_OVERLAP,
                                         lambda r, lo, hi, k: np.round(C.sample_uniform_float(r, lo, hi, k), 3))
            return {"job_title": job_title, "company_name": company_name,
                    "endorsements_count": endorsements, "skills_count": skills,
                    "recommendation_count": recommendations, "connection_acceptance_rate": acceptance}
        return build

    def fake_recruiter_builder(rng, fk, n):
        job_title = [JOB_TITLES_FAKE_RECRUITER[int(rng.integers(0, len(JOB_TITLES_FAKE_RECRUITER)))]
                     for _ in range(n)]
        company_name = [IMPLAUSIBLE_COMPANIES[int(rng.integers(0, len(IMPLAUSIBLE_COMPANIES)))]
                        if rng.random() < 0.6 else _indian_company(rng) for _ in range(n)]
        endorsements = np.round(C.sample_lognormal_clipped(rng, 0, 20, n)).astype(int)
        skills = np.round(C.sample_uniform_float(rng, 0, 8, n)).astype(int)
        endorsements = _apply_overlap(rng, endorsements, n, 1, LI_GENUINE_ENDORSEMENT_OVERLAP,
                                       LI_FAKE_ENDORSEMENT_OVERLAP,
                                       lambda r, lo, hi, k: np.round(C.sample_lognormal_clipped(r, lo, hi, k)).astype(int))
        skills = _apply_overlap(rng, skills, n, 1, LI_GENUINE_SKILLS_OVERLAP, LI_FAKE_SKILLS_OVERLAP,
                                 lambda r, lo, hi, k: np.round(C.sample_uniform_float(r, lo, hi, k)).astype(int))
        # Fake recruiters mass-send cold connection requests to source "candidates" --
        # low acceptance rate is the signature; recommendations are nearly impossible
        # to fake since real people rarely write one for a stranger.
        recommendations = np.round(C.sample_lognormal_clipped(rng, 0, 2, n)).astype(int)
        acceptance = np.round(C.sample_uniform_float(rng, 0.05, 0.3, n), 3)
        recommendations = _apply_overlap(rng, recommendations, n, 1, LI_GENUINE_RECOMMENDATION_OVERLAP,
                                          LI_FAKE_RECOMMENDATION_OVERLAP,
                                          lambda r, lo, hi, k: np.round(C.sample_lognormal_clipped(r, lo, hi, k)).astype(int))
        acceptance = _apply_overlap(rng, acceptance, n, 1, LI_GENUINE_ACCEPTANCE_OVERLAP,
                                     LI_FAKE_ACCEPTANCE_OVERLAP,
                                     lambda r, lo, hi, k: np.round(C.sample_uniform_float(r, lo, hi, k), 3))
        return {"job_title": job_title, "company_name": company_name,
                "endorsements_count": endorsements, "skills_count": skills,
                "recommendation_count": recommendations, "connection_acceptance_rate": acceptance}

    def fake_exec_builder(rng, fk, n):
        job_title = [JOB_TITLES_FAKE_EXEC[int(rng.integers(0, len(JOB_TITLES_FAKE_EXEC)))] for _ in range(n)]
        company_name = [IMPLAUSIBLE_COMPANIES[int(rng.integers(0, len(IMPLAUSIBLE_COMPANIES)))]
                        for _ in range(n)]
        endorsements = np.round(C.sample_lognormal_clipped(rng, 0, 15, n)).astype(int)
        skills = np.round(C.sample_uniform_float(rng, 0, 6, n)).astype(int)
        endorsements = _apply_overlap(rng, endorsements, n, 1, LI_GENUINE_ENDORSEMENT_OVERLAP,
                                       LI_FAKE_ENDORSEMENT_OVERLAP,
                                       lambda r, lo, hi, k: np.round(C.sample_lognormal_clipped(r, lo, hi, k)).astype(int))
        skills = _apply_overlap(rng, skills, n, 1, LI_GENUINE_SKILLS_OVERLAP, LI_FAKE_SKILLS_OVERLAP,
                                 lambda r, lo, hi, k: np.round(C.sample_uniform_float(r, lo, hi, k)).astype(int))
        recommendations = np.round(C.sample_lognormal_clipped(rng, 0, 3, n)).astype(int)
        acceptance = np.round(C.sample_uniform_float(rng, 0.05, 0.35, n), 3)
        recommendations = _apply_overlap(rng, recommendations, n, 1, LI_GENUINE_RECOMMENDATION_OVERLAP,
                                          LI_FAKE_RECOMMENDATION_OVERLAP,
                                          lambda r, lo, hi, k: np.round(C.sample_lognormal_clipped(r, lo, hi, k)).astype(int))
        acceptance = _apply_overlap(rng, acceptance, n, 1, LI_GENUINE_ACCEPTANCE_OVERLAP,
                                     LI_FAKE_ACCEPTANCE_OVERLAP,
                                     lambda r, lo, hi, k: np.round(C.sample_uniform_float(r, lo, hi, k), 3))
        return {"job_title": job_title, "company_name": company_name,
                "endorsements_count": endorsements, "skills_count": skills,
                "recommendation_count": recommendations, "connection_acceptance_rate": acceptance}

    def spam_builder(rng, fk, n):
        job_title = [JOB_TITLES_SPAM[int(rng.integers(0, len(JOB_TITLES_SPAM)))] for _ in range(n)]
        company_name = ["Self-Employed" if rng.random() < 0.7 else _indian_company(rng) for _ in range(n)]
        endorsements = np.round(C.sample_lognormal_clipped(rng, 0, 10, n)).astype(int)
        skills = np.round(C.sample_uniform_float(rng, 0, 5, n)).astype(int)
        endorsements = _apply_overlap(rng, endorsements, n, 1, LI_GENUINE_ENDORSEMENT_OVERLAP,
                                       LI_FAKE_ENDORSEMENT_OVERLAP,
                                       lambda r, lo, hi, k: np.round(C.sample_lognormal_clipped(r, lo, hi, k)).astype(int))
        skills = _apply_overlap(rng, skills, n, 1, LI_GENUINE_SKILLS_OVERLAP, LI_FAKE_SKILLS_OVERLAP,
                                 lambda r, lo, hi, k: np.round(C.sample_uniform_float(r, lo, hi, k)).astype(int))
        recommendations = np.round(C.sample_lognormal_clipped(rng, 0, 1, n)).astype(int)
        acceptance = np.round(C.sample_uniform_float(rng, 0.02, 0.2, n), 3)
        recommendations = _apply_overlap(rng, recommendations, n, 1, LI_GENUINE_RECOMMENDATION_OVERLAP,
                                          LI_FAKE_RECOMMENDATION_OVERLAP,
                                          lambda r, lo, hi, k: np.round(C.sample_lognormal_clipped(r, lo, hi, k)).astype(int))
        acceptance = _apply_overlap(rng, acceptance, n, 1, LI_GENUINE_ACCEPTANCE_OVERLAP,
                                     LI_FAKE_ACCEPTANCE_OVERLAP,
                                     lambda r, lo, hi, k: np.round(C.sample_uniform_float(r, lo, hi, k), 3))
        return {"job_title": job_title, "company_name": company_name,
                "endorsements_count": endorsements, "skills_count": skills,
                "recommendation_count": recommendations, "connection_acceptance_rate": acceptance}

    return {
        "Student": genuine_builder("Student"),
        "Developer": genuine_builder("Developer"),
        "Business/Company Page": genuine_builder("Business/Company Page"),
        "Consultant": genuine_builder("Consultant"),
        "Fake Recruiter": fake_recruiter_builder,
        "Fake Executive Impersonator": fake_exec_builder,
        "Spam/Bot Account": spam_builder,
    }


def facebook_extra_builders() -> dict[str, ExtraFieldBuilder]:
    ranges = {
        "Student": (30, 800), "Developer": (20, 600), "Business Page": (0, 50), "Fitness Creator": (20, 800),
        "Spam Account": (0, 5), "Fake Influencer": (0, 15), "Celebrity Impersonator": (0, 10),
        "Catfish/Romance-Scam Profile": (0, 3),
    }
    is_fake_by_name = {
        "Student": 0, "Developer": 0, "Business Page": 0, "Fitness Creator": 0,
        "Spam Account": 1, "Fake Influencer": 1, "Celebrity Impersonator": 1,
        "Catfish/Romance-Scam Profile": 1,
    }
    genuine_overlap_range = (0, 12)     # mutual_friends_count typical of a fake account
    fake_overlap_range = (25, 400)      # mutual_friends_count typical of a genuine account

    # A real user joins a handful of interest/community groups over time; fake
    # accounts rarely bother (no incentive) or join a suspiciously large spammy set.
    group_ranges = {
        "Student": (2, 15), "Developer": (1, 10), "Business Page": (0, 4), "Fitness Creator": (1, 8),
        "Spam Account": (0, 2), "Fake Influencer": (0, 3), "Celebrity Impersonator": (0, 2),
        "Catfish/Romance-Scam Profile": (0, 1),
    }
    genuine_group_overlap = (0, 2)    # what a fake account's group count typically looks like
    fake_group_overlap = (3, 15)      # what a genuine account's group count typically looks like

    def make(name):
        lo, hi = ranges[name]
        glo, ghi = group_ranges[name]
        is_fake = is_fake_by_name[name]

        def build(rng, fk, n):
            mutual = np.round(C.sample_lognormal_clipped(rng, lo, hi, n)).astype(int)
            n_ov = int(round(n * OVERLAP_FRAC))
            if n_ov > 0:
                idx = rng.choice(n, size=n_ov, replace=False)
                opp_range = genuine_overlap_range if is_fake == 0 else fake_overlap_range
                mutual[idx] = np.round(C.sample_lognormal_clipped(rng, *opp_range, n_ov)).astype(int)
            groups = np.round(C.sample_uniform_float(rng, glo, ghi, n)).astype(int)
            groups = _apply_overlap(rng, groups, n, is_fake, genuine_group_overlap, fake_group_overlap,
                                     lambda r, lo2, hi2, k: np.round(C.sample_uniform_float(r, lo2, hi2, k)).astype(int))
            return {"mutual_friends_count": mutual, "group_membership_count": groups}
        return build

    return {name: make(name) for name in ranges}


def instagram_extra_builders() -> dict[str, ExtraFieldBuilder]:
    # Stories are low-effort/high-frequency for genuine, engaged users; fake/spam
    # accounts optimize for feed posts and reach, not ephemeral content nobody pays for.
    ranges = {
        "Student": (1.0, 8.0), "Developer": (0.2, 4.0), "Business Account": (1.0, 10.0),
        "Fitness Creator": (2.0, 14.0), "Spam Account": (0.0, 0.5),
        "Fake Influencer": (0.0, 2.0), "Celebrity Impersonator": (0.0, 1.0),
    }
    is_fake_by_name = {
        "Student": 0, "Developer": 0, "Business Account": 0, "Fitness Creator": 0,
        "Spam Account": 1, "Fake Influencer": 1, "Celebrity Impersonator": 1,
    }
    genuine_overlap = (0.0, 1.0)   # what a fake account's story rate typically looks like
    fake_overlap = (1.5, 9.0)      # what a genuine account's story rate typically looks like

    def make(name):
        lo, hi = ranges[name]
        is_fake = is_fake_by_name[name]

        def build(rng, fk, n):
            vals = C.sample_uniform_float(rng, lo, hi, n)
            vals = _apply_overlap(rng, vals, n, is_fake, genuine_overlap, fake_overlap,
                                   lambda r, lo2, hi2, k: C.sample_uniform_float(r, lo2, hi2, k))
            return {"story_post_rate_weekly": np.round(vals, 2)}
        return build

    return {name: make(name) for name in ranges}


def twitter_extra_builders() -> dict[str, ExtraFieldBuilder]:
    # retweet_ratio = share of a profile's activity that's retweets vs. original
    # tweets. Genuine users mix both; bot/spam accounts often either retweet
    # almost nothing original (pure amplification bots) or never retweet at all
    # (pure broadcast spam) -- both ends of the range are covered per archetype.
    ranges = {
        "Student": (0.15, 0.55), "Developer": (0.2, 0.6), "Business Account": (0.05, 0.35),
        "Fitness Creator": (0.1, 0.45), "Spam/Bot Account": (0.6, 0.98),
        "Fake Influencer": (0.5, 0.9), "Celebrity Impersonator": (0.0, 0.15),
    }
    is_fake_by_name = {
        "Student": 0, "Developer": 0, "Business Account": 0, "Fitness Creator": 0,
        "Spam/Bot Account": 1, "Fake Influencer": 1, "Celebrity Impersonator": 1,
    }
    genuine_overlap = (0.45, 0.75)  # what a fake account's retweet ratio typically looks like
    fake_overlap = (0.15, 0.45)     # what a genuine account's retweet ratio typically looks like

    def make(name):
        lo, hi = ranges[name]
        is_fake = is_fake_by_name[name]

        def build(rng, fk, n):
            vals = C.sample_uniform_float(rng, lo, hi, n)
            vals = _apply_overlap(rng, vals, n, is_fake, genuine_overlap, fake_overlap,
                                   lambda r, lo2, hi2, k: C.sample_uniform_float(r, lo2, hi2, k))
            return {"retweet_ratio": np.round(vals, 3)}
        return build

    return {name: make(name) for name in ranges}


# ===========================================================================
# Orchestration
# ===========================================================================

OVERLAP_FRAC = 0.13
# Text needs a much bigger overlap fraction than numeric fields to land in a realistic
# accuracy range -- see build_platform_dataset's docstring for why (disjoint per-archetype
# vocabulary lets a lexical/semantic classifier nearly perfectly separate classes even when
# the numeric layer alone is realistically noisy).
TEXT_OVERLAP_FRAC = 0.15

# Shared platform-owner telemetry columns, appended in the same order on every
# platform (see build_platform_dataset / common.sample_trust_signals).
TRUST_COLUMN_ORDER = ["city", "state", "email_verified", "phone_verified", "two_factor_enabled",
                       "device_count_30d", "login_ip_diversity_30d", "signup_to_first_post_hours",
                       "posting_time_entropy", "follower_growth_rate_7d", "reports_received_count",
                       "content_removed_count"]
TRUST_COLUMN_SCHEMA = {
    "city": "str", "state": "str", "email_verified": "bool", "phone_verified": "bool",
    "two_factor_enabled": "bool", "device_count_30d": "int", "login_ip_diversity_30d": "int",
    "signup_to_first_post_hours": "float", "posting_time_entropy": "float",
    "follower_growth_rate_7d": "float", "reports_received_count": "int", "content_removed_count": "int",
}

PLATFORM_SPECS = {
    "ig": dict(
        platform="instagram", prefix="ig", archetypes_fn=instagram_archetypes,
        posts_count_field="posts_count", primary_field="followers_count", secondary_field="following_count",
        engagement_denominator_fields=["followers_count"], extra_field_builders=instagram_extra_builders(),
        text_field="bio",
        profile_column_order=["user_id", "username", "bio", "followers_count", "following_count", "posts_count",
                               "account_age_days", "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                               "is_verified", "has_website", "has_location", "profile_completion_score",
                               "story_post_rate_weekly", *TRUST_COLUMN_ORDER, "archetype", "is_fake"],
    ),
    "tw": dict(
        platform="twitter", prefix="tw", archetypes_fn=twitter_archetypes,
        posts_count_field="tweets_count", primary_field="followers_count", secondary_field="following_count",
        engagement_denominator_fields=["followers_count"], extra_field_builders=twitter_extra_builders(),
        text_field="bio",
        profile_column_order=["user_id", "username", "bio", "followers_count", "following_count", "tweets_count",
                               "account_age_days", "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                               "is_verified", "has_website", "has_location", "profile_completion_score",
                               "retweet_ratio", *TRUST_COLUMN_ORDER, "archetype", "is_fake"],
    ),
    "fb": dict(
        platform="facebook", prefix="fb", archetypes_fn=facebook_archetypes,
        posts_count_field="posts_count", primary_field="friends_count", secondary_field="followers_count",
        engagement_denominator_fields=["friends_count", "followers_count"],
        extra_field_builders=facebook_extra_builders(), text_field="bio",
        profile_column_order=["user_id", "username", "bio", "friends_count", "followers_count", "posts_count",
                               "account_age_days", "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                               "mutual_friends_count", "group_membership_count", "is_verified", "has_website",
                               "has_location", "profile_completion_score", *TRUST_COLUMN_ORDER,
                               "archetype", "is_fake"],
    ),
    "li": dict(
        platform="linkedin", prefix="li", archetypes_fn=linkedin_archetypes,
        posts_count_field="posts_count", primary_field="connections_count", secondary_field=None,
        engagement_denominator_fields=["connections_count"],
        extra_field_builders=linkedin_extra_builders(), text_field="headline",
        profile_column_order=["user_id", "username", "headline", "connections_count", "posts_count",
                               "account_age_days", "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                               "job_title", "company_name", "endorsements_count", "skills_count",
                               "recommendation_count", "connection_acceptance_rate",
                               "is_verified", "has_website", "has_location", "profile_completion_score",
                               *TRUST_COLUMN_ORDER, "archetype", "is_fake"],
    ),
}

PROFILE_SCHEMA = {
    "ig": {"user_id": "str", "username": "str", "bio": "str", "followers_count": "int", "following_count": "int",
           "posts_count": "int", "account_age_days": "int", "avg_likes": "float", "avg_comments": "float",
           "avg_shares": "float", "engagement_rate": "float", "is_verified": "bool", "has_website": "bool",
           "has_location": "bool", "profile_completion_score": "float", "story_post_rate_weekly": "float",
           **TRUST_COLUMN_SCHEMA, "archetype": "str", "is_fake": "int"},
    "tw": {"user_id": "str", "username": "str", "bio": "str", "followers_count": "int", "following_count": "int",
           "tweets_count": "int", "account_age_days": "int", "avg_likes": "float", "avg_comments": "float",
           "avg_shares": "float", "engagement_rate": "float", "is_verified": "bool", "has_website": "bool",
           "has_location": "bool", "profile_completion_score": "float", "retweet_ratio": "float",
           **TRUST_COLUMN_SCHEMA, "archetype": "str", "is_fake": "int"},
    "fb": {"user_id": "str", "username": "str", "bio": "str", "friends_count": "int", "followers_count": "int",
           "posts_count": "int", "account_age_days": "int", "avg_likes": "float", "avg_comments": "float",
           "avg_shares": "float", "engagement_rate": "float", "mutual_friends_count": "int",
           "group_membership_count": "int", "is_verified": "bool",
           "has_website": "bool", "has_location": "bool", "profile_completion_score": "float",
           **TRUST_COLUMN_SCHEMA, "archetype": "str", "is_fake": "int"},
    "li": {"user_id": "str", "username": "str", "headline": "str", "connections_count": "int", "posts_count": "int",
           "account_age_days": "int", "avg_likes": "float", "avg_comments": "float", "avg_shares": "float",
           "engagement_rate": "float", "job_title": "str", "company_name": "str", "endorsements_count": "int",
           "skills_count": "int", "recommendation_count": "int", "connection_acceptance_rate": "float",
           "is_verified": "bool", "has_website": "bool", "has_location": "bool",
           "profile_completion_score": "float", **TRUST_COLUMN_SCHEMA, "archetype": "str", "is_fake": "int"},
}

POST_SCHEMA = {
    "user_id": "str", "caption": "str", "hashtags": "str", "likes": "int", "comments": "int", "shares": "int",
    "sentiment": "float", "archetype": "str", "is_fake": "int",
}

NULLABLE_COLS = ["hashtags"]  # Facebook/LinkedIn hashtags legitimately empty strings, never null


def run_platform(key: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], bool]:
    spec = PLATFORM_SPECS[key]
    archetypes = spec["archetypes_fn"]()
    profiles, posts = build_platform_dataset(
        platform=spec["platform"], prefix=spec["prefix"], archetypes=archetypes,
        n_profiles=C.TARGET_PROFILES_PER_PLATFORM, posts_count_field=spec["posts_count_field"],
        primary_field=spec["primary_field"], secondary_field=spec["secondary_field"],
        engagement_denominator_fields=spec["engagement_denominator_fields"],
        extra_field_builders=spec["extra_field_builders"], profile_column_order=spec["profile_column_order"],
        text_field=spec["text_field"], overlap_frac=OVERLAP_FRAC, text_overlap_frac=TEXT_OVERLAP_FRAC,
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    profiles.to_csv(os.path.join(DATA_DIR, f"{spec['platform']}_profiles.csv"), index=False)
    posts.to_csv(os.path.join(DATA_DIR, f"{spec['platform']}_posts.csv"), index=False)

    report, ok = C.validate_platform(
        spec["platform"], profiles, posts, PROFILE_SCHEMA[key], POST_SCHEMA, nullable_cols=NULLABLE_COLS,
    )
    return profiles, posts, report, ok


def main():
    parser = argparse.ArgumentParser(description="Generate IDGuardian synthetic datasets.")
    parser.add_argument("--platform", choices=["ig", "fb", "li", "tw", "all"], default="all")
    args = parser.parse_args()

    keys = ["ig", "fb", "li", "tw"] if args.platform == "all" else [args.platform]

    all_report_lines: list[str] = []
    all_ok = True
    for key in keys:
        print(f"Generating {PLATFORM_SPECS[key]['platform']}...")
        profiles, posts, report, ok = run_platform(key)
        all_report_lines.extend(report)
        all_ok = all_ok and ok
        print(f"  profiles={len(profiles)}  posts={len(posts)}  validation={'OK' if ok else 'FAILED'}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "validation_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"IDGuardian synthetic dataset validation report (SEED={C.SEED})\n")
        f.write(f"Overall result: {'ALL PLATFORMS PASSED' if all_ok else 'SOME CHECKS FAILED'}\n\n")
        f.write("\n".join(all_report_lines))

    print("\n".join(all_report_lines))
    print(f"\nOverall: {'ALL PLATFORMS PASSED' if all_ok else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
