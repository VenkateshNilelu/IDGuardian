"""
IDGuardian synthetic dataset generation — shared helpers.

Everything here is platform-agnostic: RNG setup, text recombination,
numeric distribution sampling, sentiment scoring, id generation, and the
cross-cutting validation pass. Platform-specific archetype configuration
(phrase banks + numeric parameters) lives in generate_dataset.py.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from faker import Faker
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SEED = 42
TARGET_PROFILES_PER_PLATFORM = 5000
TARGET_POSTS_PER_PLATFORM = 50000
POSTS_PER_PROFILE_LAMBDA = 10.0
GENUINE_SHARE = 0.6
FAKE_SHARE = 0.4
AGGREGATE_TOLERANCE = 1e-6  # profile aggregates are computed directly from posts, so exact

_vader = SentimentIntensityAnalyzer()


def _stable_salt(salt: str) -> int:
    """Deterministic string->int hash. Python's builtin hash() is randomized
    per-process (PYTHONHASHSEED) for security, which would silently break
    reproducibility across runs -- crc32 is stable across processes/machines.
    """
    return zlib.crc32(salt.encode("utf-8"))


def make_rng(salt: str) -> np.random.Generator:
    """Deterministic, per-platform-independent RNG derived from the global SEED.

    Using a salted seed per platform means each platform's stream is
    reproducible on its own and does not silently shift if another
    platform's generation code changes upstream of it.
    """
    salted = SEED ^ _stable_salt(salt)
    return np.random.default_rng(salted)


def make_faker(salt: str) -> Faker:
    fk = Faker()
    Faker.seed(SEED ^ _stable_salt(salt))
    return fk


def sentiment_score(text: str) -> float:
    """VADER compound sentiment score in [-1, 1], deterministic given text."""
    if not text:
        return 0.0
    return float(_vader.polarity_scores(text)["compound"])


def make_user_ids(prefix: str, n: int) -> list[str]:
    width = max(6, len(str(n)))
    return [f"{prefix}_{i:0{width}d}" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Text recombination
# ---------------------------------------------------------------------------

def pick_some(rng: np.random.Generator, pool: Sequence[str], lo: int = 1, hi: int = 3) -> list[str]:
    """Pick between lo and hi (inclusive) distinct items from pool, order randomized."""
    hi = min(hi, len(pool))
    lo = min(lo, hi)
    k = int(rng.integers(lo, hi + 1))
    idx = rng.choice(len(pool), size=k, replace=False)
    items = [pool[i] for i in idx]
    rng.shuffle(items)
    return items


def maybe_emoji(rng: np.random.Generator, emoji_pool: Sequence[str], p: float = 0.5) -> str:
    if emoji_pool and rng.random() < p:
        n = int(rng.integers(1, 3))
        chosen = rng.choice(emoji_pool, size=min(n, len(emoji_pool)), replace=False)
        return " " + "".join(chosen)
    return ""


def fill_template(rng: np.random.Generator, fk: Faker, template: str) -> str:
    """Fill {name}/{first}/{company}/{school}/{city}/{number} placeholders."""
    out = template
    if "{name}" in out:
        out = out.replace("{name}", fk.first_name())
    if "{first}" in out:
        out = out.replace("{first}", fk.first_name())
    if "{company}" in out:
        out = out.replace("{company}", fk.company())
    if "{school}" in out:
        out = out.replace("{school}", f"{fk.city()} University")
    if "{city}" in out:
        out = out.replace("{city}", fk.city())
    if "{number}" in out:
        out = out.replace("{number}", str(int(rng.integers(2, 99))))
    if "{year}" in out:
        out = out.replace("{year}", str(int(rng.integers(2015, 2026))))
    if "{handle}" in out:
        out = out.replace("{handle}", "@" + fk.user_name())
    return out


@dataclass
class PhraseBank:
    """A recombination bank for one archetype's short-form text (bio/headline)."""
    fragments: dict[str, list[str]]
    emoji_pool: list[str] = field(default_factory=list)
    emoji_prob: float = 0.4
    connector: str = " "
    parts_range: tuple[int, int] = (2, 3)

    def sample(self, rng: np.random.Generator, fk: Faker) -> str:
        keys = list(self.fragments.keys())
        lo, hi = self.parts_range
        hi = min(hi, len(keys))
        lo = min(lo, hi)
        n_keys = int(rng.integers(lo, hi + 1))
        chosen_keys = list(rng.choice(keys, size=n_keys, replace=False))
        rng.shuffle(chosen_keys)
        pieces = []
        for k in chosen_keys:
            pool = self.fragments[k]
            idx = int(rng.integers(0, len(pool)))
            pieces.append(fill_template(rng, fk, pool[idx]))
        text = self.connector.join(pieces)
        text += maybe_emoji(rng, self.emoji_pool, self.emoji_prob)
        return text.strip()


@dataclass
class CaptionBank:
    """A recombination bank for post captions."""
    openers: list[str]
    bodies: list[str]
    ctas: list[str] = field(default_factory=list)
    emoji_pool: list[str] = field(default_factory=list)
    emoji_prob: float = 0.5
    cta_prob: float = 0.3

    def sample(self, rng: np.random.Generator, fk: Faker) -> str:
        parts = [
            fill_template(rng, fk, self.openers[int(rng.integers(0, len(self.openers)))]),
            fill_template(rng, fk, self.bodies[int(rng.integers(0, len(self.bodies)))]),
        ]
        if self.ctas and rng.random() < self.cta_prob:
            parts.append(fill_template(rng, fk, self.ctas[int(rng.integers(0, len(self.ctas)))]))
        rng.shuffle(parts) if False else None  # keep opener-first, natural reading order
        text = " ".join(parts)
        text += maybe_emoji(rng, self.emoji_pool, self.emoji_prob)
        return text.strip()


def sample_hashtags(rng: np.random.Generator, pool: Sequence[str], lo: int, hi: int, empty_prob: float = 0.0) -> str:
    if pool and rng.random() < empty_prob:
        return ""
    if not pool or hi <= 0:
        return ""
    hi = min(hi, len(pool))
    lo = min(lo, hi)
    k = int(rng.integers(lo, hi + 1)) if hi > 0 else 0
    if k <= 0:
        return ""
    idx = rng.choice(len(pool), size=k, replace=False)
    tags = [f"#{pool[i]}" for i in idx]
    rng.shuffle(tags)
    return " ".join(tags)


# ---------------------------------------------------------------------------
# Numeric sampling helpers
# ---------------------------------------------------------------------------

def sample_lognormal_clipped(rng: np.random.Generator, low: float, high: float, size: int,
                              skew: float = 0.6) -> np.ndarray:
    """Lognormal-shaped sample clipped into [low, high], right-skewed like real count data.

    The lognormal's median is anchored at the geometric mean of (low, high)
    rather than at low itself, so samples actually spread across the given
    range (anchoring at low would otherwise cluster nearly everything near
    the floor regardless of how large high is).
    """
    typical = np.sqrt(max(low, 1.0) * max(high, 1.0))
    mu = np.log(max(typical, 1.0))
    vals = rng.lognormal(mean=mu, sigma=skew, size=size)
    vals = np.clip(vals, low, high)
    return vals


def sample_uniform_int(rng: np.random.Generator, low: int, high: int, size: int) -> np.ndarray:
    high = max(low, high)
    return rng.integers(low, high + 1, size=size)


def sample_poisson_min1(rng: np.random.Generator, lam: float, size: int) -> np.ndarray:
    vals = rng.poisson(lam=lam, size=size)
    return np.clip(vals, 1, None)


def sample_bool(rng: np.random.Generator, p: float, size: int) -> np.ndarray:
    return rng.random(size) < p


def sample_uniform_float(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    return rng.uniform(low, high, size)


# ---------------------------------------------------------------------------
# Shared post generation + aggregate derivation
# ---------------------------------------------------------------------------

def generate_posts_for_all_profiles(
    rng: np.random.Generator,
    fk: Faker,
    profile_ids: Sequence[str],
    archetypes: Sequence[str],
    is_fake_flags: Sequence[int],
    n_posts: Sequence[int],
    audience: Sequence[float],
    engagement_rate_base: Sequence[float],
    comment_ratio: Sequence[float],
    share_ratio: Sequence[float],
    post_noise_sigma: Sequence[float],
    caption_banks: dict[str, "CaptionBank"],
    hashtag_cfg: dict[str, tuple[list[str], int, int, float]],
) -> pd.DataFrame:
    """Generate every post for every profile in one pass.

    hashtag_cfg maps archetype -> (pool, lo, hi, empty_prob).
    Engagement fields are sampled per-post around each profile's own
    archetype-conditioned base rate/ratio, with lognormal multiplicative
    noise so no two posts of the same profile look identical.
    """
    user_id_col: list[str] = []
    caption_col: list[str] = []
    hashtag_col: list[str] = []
    likes_col: list[int] = []
    comments_col: list[int] = []
    shares_col: list[int] = []
    sentiment_col: list[float] = []
    archetype_col: list[str] = []
    is_fake_col: list[int] = []

    for i in range(len(profile_ids)):
        n = int(n_posts[i])
        if n <= 0:
            continue
        arch = archetypes[i]
        bank = caption_banks[arch]
        pool, hlo, hhi, hempty = hashtag_cfg[arch]

        base_likes = max(audience[i], 1.0) * engagement_rate_base[i]
        like_noise = rng.lognormal(mean=0.0, sigma=post_noise_sigma[i], size=n)
        likes = np.round(base_likes * like_noise)
        likes = np.clip(likes, 0, None)

        comment_noise = rng.lognormal(mean=0.0, sigma=post_noise_sigma[i], size=n)
        comments = np.round(likes * comment_ratio[i] * comment_noise)
        comments = np.clip(comments, 0, None)

        share_noise = rng.lognormal(mean=0.0, sigma=post_noise_sigma[i], size=n)
        shares = np.round(likes * share_ratio[i] * share_noise)
        shares = np.clip(shares, 0, None)

        for j in range(n):
            caption = bank.sample(rng, fk)
            hashtags = sample_hashtags(rng, pool, hlo, hhi, hempty)
            user_id_col.append(profile_ids[i])
            caption_col.append(caption)
            hashtag_col.append(hashtags)
            likes_col.append(int(likes[j]))
            comments_col.append(int(comments[j]))
            shares_col.append(int(shares[j]))
            sentiment_col.append(sentiment_score(caption))
            archetype_col.append(arch)
            is_fake_col.append(int(is_fake_flags[i]))

    return pd.DataFrame({
        "user_id": user_id_col,
        "caption": caption_col,
        "hashtags": hashtag_col,
        "likes": pd.array(likes_col, dtype="int64"),
        "comments": pd.array(comments_col, dtype="int64"),
        "shares": pd.array(shares_col, dtype="int64"),
        "sentiment": pd.array(sentiment_col, dtype="float64"),
        "archetype": archetype_col,
        "is_fake": pd.array(is_fake_col, dtype="int64"),
    })


def derive_profile_aggregates(posts: pd.DataFrame, profile_ids: Sequence[str]) -> pd.DataFrame:
    """Compute avg_likes/avg_comments/avg_shares/posts_count from actual posts.

    Every id in profile_ids gets a row (0-filled if somehow it had zero posts,
    though generation guarantees >=1 post per profile).
    """
    agg = posts.groupby("user_id").agg(
        avg_likes=("likes", "mean"),
        avg_comments=("comments", "mean"),
        avg_shares=("shares", "mean"),
        posts_count=("likes", "size"),
    ).reindex(profile_ids)
    agg["avg_likes"] = agg["avg_likes"].fillna(0.0)
    agg["avg_comments"] = agg["avg_comments"].fillna(0.0)
    agg["avg_shares"] = agg["avg_shares"].fillna(0.0)
    agg["posts_count"] = agg["posts_count"].fillna(0).astype("int64")
    agg = agg.reset_index().rename(columns={"index": "user_id"})
    return agg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_platform(
    platform: str,
    profiles: pd.DataFrame,
    posts: pd.DataFrame,
    profile_schema: dict[str, str],
    post_schema: dict[str, str],
    nullable_cols: Sequence[str] = (),
    tolerance: float = 1e-3,
) -> tuple[list[str], bool]:
    """Run the cross-cutting-rule-9 validation checks for one platform.

    Returns (report_lines, all_ok).
    """
    lines: list[str] = []
    ok = True

    def log(msg: str, passed: bool | None = None):
        nonlocal ok
        if passed is False:
            ok = False
        lines.append(msg)

    lines.append(f"=== {platform.upper()} VALIDATION REPORT ===")

    # 1. Orphan posts (every post.user_id must exist in profiles.user_id)
    profile_ids = set(profiles["user_id"])
    orphan_mask = ~posts["user_id"].isin(profile_ids)
    n_orphans = int(orphan_mask.sum())
    log(f"[{'PASS' if n_orphans == 0 else 'FAIL'}] Orphan posts (user_id not in profiles): {n_orphans}",
        n_orphans == 0)

    # 2. Label consistency (archetype + is_fake match parent profile)
    merged = posts.merge(
        profiles[["user_id", "archetype", "is_fake"]],
        on="user_id", how="left", suffixes=("_post", "_profile")
    )
    archetype_mismatch = int((merged["archetype_post"] != merged["archetype_profile"]).sum())
    isfake_mismatch = int((merged["is_fake_post"] != merged["is_fake_profile"]).sum())
    log(f"[{'PASS' if archetype_mismatch == 0 else 'FAIL'}] Post/profile archetype mismatches: {archetype_mismatch}",
        archetype_mismatch == 0)
    log(f"[{'PASS' if isfake_mismatch == 0 else 'FAIL'}] Post/profile is_fake mismatches: {isfake_mismatch}",
        isfake_mismatch == 0)

    # 3. Aggregate consistency: profile avg_* must equal mean of that profile's posts
    post_agg = posts.groupby("user_id").agg(
        computed_avg_likes=("likes", "mean"),
        computed_avg_comments=("comments", "mean"),
        computed_avg_shares=("shares", "mean"),
        computed_posts_count=("likes", "size"),
    ).reset_index()
    check = profiles.merge(post_agg, on="user_id", how="left")
    for col, computed_col in [
        ("avg_likes", "computed_avg_likes"),
        ("avg_comments", "computed_avg_comments"),
        ("avg_shares", "computed_avg_shares"),
    ]:
        diff = (check[col] - check[computed_col]).abs()
        n_bad = int((diff > tolerance).sum())
        log(f"[{'PASS' if n_bad == 0 else 'FAIL'}] Profile {col} matches mean of linked posts "
            f"(tol={tolerance}): {n_bad} mismatches", n_bad == 0)
    posts_count_col = "posts_count" if "posts_count" in profiles.columns else "tweets_count"
    n_bad_count = int((check[posts_count_col] != check["computed_posts_count"]).sum())
    log(f"[{'PASS' if n_bad_count == 0 else 'FAIL'}] Profile {posts_count_col} matches actual linked post "
        f"count: {n_bad_count} mismatches", n_bad_count == 0)

    # 4. Per-archetype counts + overall fake/genuine balance
    lines.append("-- Archetype distribution (profiles) --")
    for arch, cnt in profiles["archetype"].value_counts().sort_index().items():
        lines.append(f"    {arch}: {cnt}")
    overall_fake_pct = 100.0 * profiles["is_fake"].mean()
    lines.append(f"-- Overall balance: {100 - overall_fake_pct:.1f}% genuine / {overall_fake_pct:.1f}% fake "
                 f"(target ~60/40)")
    lines.append("-- Archetype distribution (posts) --")
    for arch, cnt in posts["archetype"].value_counts().sort_index().items():
        lines.append(f"    {arch}: {cnt}")

    # 5. Schema / dtype / null check
    def dtype_ok(series: pd.Series, kind: str) -> bool:
        dt = series.dtype
        if kind == "str":
            return dt == object or pd.api.types.is_string_dtype(dt)
        if kind == "int":
            return pd.api.types.is_integer_dtype(dt)
        if kind == "float":
            return pd.api.types.is_float_dtype(dt)
        if kind == "bool":
            return pd.api.types.is_bool_dtype(dt)
        return True

    lines.append("-- Schema check: profiles --")
    missing_cols = [c for c in profile_schema if c not in profiles.columns]
    extra_cols = [c for c in profiles.columns if c not in profile_schema]
    log(f"[{'PASS' if not missing_cols and not extra_cols else 'FAIL'}] Profile columns match spec "
        f"(missing={missing_cols}, extra={extra_cols})", not missing_cols and not extra_cols)
    for col, dtype_kind in profile_schema.items():
        if col not in profiles.columns:
            continue
        n_null = int(profiles[col].isna().sum())
        allowed_null = col in nullable_cols
        bad_null = n_null > 0 and not allowed_null
        bad_dtype = not dtype_ok(profiles[col], dtype_kind)
        status = "PASS" if not bad_null and not bad_dtype else "FAIL"
        log(f"    [{status}] {col} ({dtype_kind}, actual={profiles[col].dtype}): nulls={n_null}",
            not bad_null and not bad_dtype)

    lines.append("-- Schema check: posts --")
    missing_cols_p = [c for c in post_schema if c not in posts.columns]
    extra_cols_p = [c for c in posts.columns if c not in post_schema]
    log(f"[{'PASS' if not missing_cols_p and not extra_cols_p else 'FAIL'}] Post columns match spec "
        f"(missing={missing_cols_p}, extra={extra_cols_p})", not missing_cols_p and not extra_cols_p)
    for col, dtype_kind in post_schema.items():
        if col not in posts.columns:
            continue
        n_null = int(posts[col].isna().sum())
        allowed_null = col in nullable_cols
        bad_null = n_null > 0 and not allowed_null
        bad_dtype = not dtype_ok(posts[col], dtype_kind)
        status = "PASS" if not bad_null and not bad_dtype else "FAIL"
        log(f"    [{status}] {col} ({dtype_kind}, actual={posts[col].dtype}): nulls={n_null}",
            not bad_null and not bad_dtype)

    lines.append(f"=== {platform.upper()}: {'ALL CHECKS PASSED' if ok else 'ONE OR MORE CHECKS FAILED'} ===\n")
    return lines, ok
