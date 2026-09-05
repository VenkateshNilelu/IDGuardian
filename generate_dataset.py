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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = C.make_rng(platform)
    fk = C.make_faker(platform)

    genuine = [a for a in archetypes if a.is_fake == 0]
    fake = [a for a in archetypes if a.is_fake == 1]
    n_genuine = round(n_profiles * C.GENUINE_SHARE)
    n_fake = n_profiles - n_genuine
    genuine_counts = even_split(n_genuine, len(genuine))
    fake_counts = even_split(n_fake, len(fake))
    blocks = list(zip(genuine, genuine_counts)) + list(zip(fake, fake_counts))

    cols: dict[str, list] = {
        "text": [], "username": [], "archetype": [], "is_fake": [],
        "account_age_days": [], "is_verified": [], "has_website": [], "has_location": [],
        "profile_completion_score": [], "primary": [], "secondary": [],
        "engagement_rate_base": [], "comment_ratio": [], "share_ratio": [],
        "post_noise_sigma": [], "n_posts": [],
    }
    extra_accum: dict[str, list[np.ndarray]] = {}
    caption_banks: dict[str, C.CaptionBank] = {}
    hashtag_cfg: dict[str, tuple] = {}

    for arche, cnt in blocks:
        caption_banks[arche.name] = arche.caption_bank
        hashtag_cfg[arche.name] = (arche.hashtag_pool, *arche.hashtag_range, arche.hashtag_empty_prob)
        if cnt <= 0:
            continue

        cols["text"].extend(arche.text_bank.sample(rng, fk) for _ in range(cnt))
        cols["username"].extend(make_username(rng, fk, arche) for _ in range(cnt))
        cols["archetype"].extend([arche.name] * cnt)
        cols["is_fake"].extend([arche.is_fake] * cnt)
        cols["account_age_days"].extend(C.sample_uniform_int(rng, *arche.account_age_range, cnt).tolist())
        cols["is_verified"].extend(C.sample_bool(rng, arche.is_verified_p, cnt).tolist())
        cols["has_website"].extend(C.sample_bool(rng, arche.has_website_p, cnt).tolist())
        cols["has_location"].extend(C.sample_bool(rng, arche.has_location_p, cnt).tolist())
        cols["profile_completion_score"].extend(
            np.round(C.sample_uniform_float(rng, *arche.profile_completion_range, cnt), 3).tolist()
        )

        primary = np.round(C.sample_lognormal_clipped(rng, *arche.primary_range, cnt)).astype(int)
        cols["primary"].extend(primary.tolist())

        if arche.secondary_mode == "ratio_near_1":
            mult = C.sample_uniform_float(rng, *arche.ratio_range, cnt)
            secondary = np.round(primary * mult).clip(min=1).astype(int)
        elif arche.secondary_mode == "independent":
            secondary = np.round(C.sample_lognormal_clipped(rng, *arche.secondary_range, cnt)).astype(int)
        else:
            secondary = np.zeros(cnt, dtype=int)
        cols["secondary"].extend(secondary.tolist())

        cols["engagement_rate_base"].extend(C.sample_uniform_float(rng, *arche.engagement_rate_range, cnt).tolist())
        cols["comment_ratio"].extend(C.sample_uniform_float(rng, *arche.comment_ratio_range, cnt).tolist())
        cols["share_ratio"].extend(C.sample_uniform_float(rng, *arche.share_ratio_range, cnt).tolist())
        cols["post_noise_sigma"].extend(C.sample_uniform_float(rng, *arche.post_noise_sigma_range, cnt).tolist())
        cols["n_posts"].extend(C.sample_poisson_min1(rng, C.POSTS_PER_PROFILE_LAMBDA, cnt).tolist())

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
    )

    agg = C.derive_profile_aggregates(posts, user_ids.tolist())

    profile_data = {
        "user_id": user_ids,
        "username": P("username"),
        primary_field: primary_arr,
        "account_age_days": Pnum("account_age_days", "int64"),
        "is_verified": Pnum("is_verified", "bool"),
        "has_website": Pnum("has_website", "bool"),
        "has_location": Pnum("has_location", "bool"),
        "profile_completion_score": Pnum("profile_completion_score", "float64"),
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
            "vibe": ["living on iced coffee and hope", "counting down to graduation",
                     "here for the memes and the free pizza", "just trying to pass orgo"],
        },
        emoji_pool=EMOJI_CASUAL, emoji_prob=0.5, parts_range=(2, 3),
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
    )


def bank_business(kind: str = "Account") -> C.PhraseBank:
    return C.PhraseBank(
        fragments={
            "role": [f"{{company}} — official {kind.lower()}", "Family-owned, {city}-based",
                     "Serving {city} since {year}", "Small business, big heart",
                     "Handmade goods from {city}", "Local favorite in {city}"],
            "offer": ["Custom orders welcome", "DM us for inquiries", "Free shipping over $50",
                      "New drops every {number} weeks", "Booking now for {year}",
                      "Quality you can trust"],
            "vibe": ["Proudly independent", "Woman-owned & operated", "Community first, always",
                     "Thank you for supporting small business"],
        },
        emoji_pool=EMOJI_BIZ, emoji_prob=0.3, parts_range=(2, 3),
    )


def bank_fitness() -> C.PhraseBank:
    return C.PhraseBank(
        fragments={
            "role": ["Certified personal trainer", "Online coach @ {company}", "Powerlifter & coach",
                     "Helping you build strength, not just muscle", "{first} — movement is medicine",
                     "Marathoner turned strength coach"],
            "focus": ["macro-friendly recipes on the blog", "form check requests welcome",
                      "5am workouts, no excuses", "mobility nerd", "believer in progressive overload",
                      "recovering cardio-phobe turned lifter"],
            "cta": ["Programs linked below", "DM 'START' for coaching info", "Free workout guide in bio",
                    "Booking 1:1 sessions"],
        },
        emoji_pool=EMOJI_FIT, emoji_prob=0.6, parts_range=(2, 3),
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
    )


def cap_generic(openers, bodies, ctas, emoji_pool, cta_prob=0.3, emoji_prob=0.5) -> C.CaptionBank:
    return C.CaptionBank(openers=openers, bodies=bodies, ctas=ctas, emoji_pool=emoji_pool,
                          cta_prob=cta_prob, emoji_prob=emoji_prob)


CAP_STUDENT = cap_generic(
    openers=["Another day surviving {school}.", "Finals season has me questioning everything.",
             "Group project update:", "Campus is so pretty in {city} today.", "Coffee #{number} of the day.",
             "Study session with {first} turned into a nap."],
    bodies=["Somehow still passing my classes.", "My professor just dropped a pop quiz, send help.",
            "Library until it closes, again.", "Ranking my classes from bearable to soul-crushing.",
            "Turns out procrastination is a full-time job.", "Dorm life really tests your patience."],
    ctas=["Send snacks.", "Wish me luck.", "Tell me it gets easier.", "Someone quiz me please."],
    emoji_pool=EMOJI_CASUAL,
)

CAP_DEVELOPER = cap_generic(
    openers=["Spent way too long on a bug today.", "Shipped a small feature at {company}.",
             "Refactoring old code I wrote a year ago.", "New side project idea just hit me at 2am.",
             "Finally got the tests passing.", "Learning a new part of the stack this week."],
    bodies=["Turned out to be a missing semicolon.", "The code review comments humbled me.",
            "Past me really did not comment anything.", "CI is green and I'm emotionally fine now.",
            "Docs were wrong but figured it out eventually.", "Pairing with a teammate made it so much faster."],
    ctas=["Repo link below.", "Curious how others solve this.", "Open to feedback.", "AMA about the stack."],
    emoji_pool=EMOJI_DEV,
)

CAP_BUSINESS = cap_generic(
    openers=["New arrivals just dropped!", "Thank you {city} for another great week.",
             "Behind the scenes at the shop today.", "Restocked your favorites.",
             "Meet the team making it happen.", "Small batch, made with care."],
    bodies=["We appreciate every single one of you.", "Orders ship out within 2 business days.",
            "Handmade in small batches every week.", "Your support keeps our doors open.",
            "Custom requests are always welcome.", "Locally sourced, always."],
    ctas=["Shop the link in bio.", "DM us to order.", "Tag someone who'd love this.", "Limited stock available."],
    emoji_pool=EMOJI_BIZ, cta_prob=0.5,
)

CAP_FITNESS = cap_generic(
    openers=["Leg day recap.", "New PR today!", "Meal prep Sunday.", "Recovery day, mobility work only.",
             "Coaching client hit a big milestone.", "Early morning session before work."],
    bodies=["Form over ego, always.", "Consistency beats intensity most days.",
            "Progress isn't always linear, and that's fine.", "Fueling properly made a huge difference.",
            "Small wins add up over months.", "Rest days are still training days."],
    ctas=["Programs linked in bio.", "DM 'START' to work together.", "Save this for your next session.",
          "Tag your gym partner."],
    emoji_pool=EMOJI_FIT, cta_prob=0.5,
)

CAP_SPAM = cap_generic(
    openers=["🚨 LAST CHANCE 🚨", "Everyone is asking how I did this so fast.", "Stop scrolling, read this.",
             "I wasn't going to share this but...", "This changed everything for me.", "You NEED to see this."],
    bodies=["Made more money this week than my old job paid monthly.", "It's 100% legit, I was skeptical too.",
            "Spots are filling up fast.", "No experience needed, just a phone.", "This is not a scam, I promise.",
            "Thousands already joined, don't miss out."],
    ctas=["Link in bio, click now.", "DM me 'START' immediately.", "Comment before it's gone.",
          "Tap the link in my story."],
    emoji_pool=EMOJI_SPAM, cta_prob=0.8, emoji_prob=0.8,
)

CAP_FAKE_INFLUENCER = cap_generic(
    openers=["Obsessed with this new find.", "Another day, another shoot.", "This brand sent me the best package.",
             "Golden hour never disappoints.", "Living for moments like this.", "Can't stop wearing this lately."],
    bodies=["Use my code for a discount.", "So grateful for opportunities like this.",
            "This is exactly what my feed needed.", "Can't wait to show you more from this collab.",
            "My followers deserve the best recommendations.", "Partnering with brands I actually love."],
    ctas=["Link in bio for the discount.", "Code is my username at checkout.", "Shop this look now.",
          "DM for collab details."],
    emoji_pool=EMOJI_GLAM, cta_prob=0.6,
)

CAP_CELEBRITY_IMPERSONATOR = cap_generic(
    openers=["To my real fans:", "Thank you for {number} years of love.", "Important announcement coming soon.",
             "I don't get to say this enough.", "This account is really me, I promise."],
    bodies=["Please report any fake pages you see.", "I read every message even if I can't reply to all.",
            "New project details coming very soon.", "Grateful for every one of you every single day.",
            "Ignore anyone claiming to be my manager elsewhere."],
    ctas=["DM me directly here.", "Follow only this page.", "Share this so others know it's real."],
    emoji_pool=EMOJI_CASUAL, cta_prob=0.4,
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
    )


CAP_CATFISH = cap_generic(
    openers=["Thinking about you today.", "Another lonely night here overseas.", "Can't wait to finally meet.",
             "Just need someone to talk to.", "Missing having someone real in my life."],
    bodies=["Work keeps me so busy but you're always on my mind.", "I don't trust easily but you're different.",
            "Money's been tight since the last assignment.", "Hoping we can talk more privately soon.",
            "I promise I'm not like the others you've met online."],
    ctas=["Message me on here.", "Let's move the conversation elsewhere.", "Please don't give up on us."],
    emoji_pool=["❤️", "🌹", "🙏"], cta_prob=0.4,
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
                 "Reflecting on my first year at {school}."],
        bodies=["Learned so much about teamwork and deadlines.", "Met some incredible recruiters and alumni.",
                "This confirmed the career path I want to pursue.", "Time to recharge before next semester.",
                "Grateful for professors who go above and beyond."],
        ctas=["Open to internship opportunities.", "Would love to connect with others in the field.",
              "Feel free to reach out!"],
        emoji_pool=[], cta_prob=0.4, emoji_prob=0.05,
    )
    cap_dev_li = cap_generic(
        openers=["Shipped a new feature at {company} this week.", "Wrote a short post on lessons from a recent outage.",
                 "Excited to start using a new framework on our team.", "Reflecting on 3 years as a developer.",
                 "Mentored a junior engineer today and loved it."],
        bodies=["Documentation really does save future you.", "Cross-team collaboration made this so much smoother.",
                "Testing early saved us a painful rollback.", "Grateful for a team that values code quality.",
                "Always learning something new in this role."],
        ctas=["Happy to share more details if useful.", "Open to connecting with other engineers.",
              "Let me know your thoughts below."],
        emoji_pool=[], cta_prob=0.3, emoji_prob=0.05,
    )
    cap_biz_li = cap_generic(
        openers=["Proud to announce a new partnership.", "Our team just hit a major milestone.",
                 "Thank you to our clients for another great quarter.", "Excited to share our latest case study.",
                 "We're hiring! Come join our growing team."],
        bodies=["This wouldn't be possible without our incredible team.", "Looking forward to what's next for us.",
                "Grateful for the trust our clients place in us.", "Innovation continues to drive everything we do.",
                "We're expanding into new markets this year."],
        ctas=["Learn more on our website.", "Reach out if you'd like to collaborate.",
              "Apply via the link in our profile."],
        emoji_pool=[], cta_prob=0.5, emoji_prob=0.02,
    )
    cap_consultant_li = cap_generic(
        openers=["Wrapped up an engagement with a great client this week.", "Some thoughts on change management.",
                 "Just published a short case study.", "Speaking at a panel next month on industry trends.",
                 "Reflecting on 10 years of consulting."],
        bodies=["Strategy is only as good as its execution.", "Every client engagement teaches me something new.",
                "The best solutions come from listening first.", "Grateful to work across such varied industries.",
                "Frameworks help, but context always wins."],
        ctas=["Happy to discuss further, DM me.", "Would love your thoughts in the comments.",
              "Reach out if this resonates."],
        emoji_pool=[], cta_prob=0.35, emoji_prob=0.02,
    )
    cap_fake_recruiter = cap_generic(
        openers=["URGENT HIRING: multiple positions open now!", "We are looking for talent, apply immediately!",
                 "Exciting remote opportunity, no experience needed!", "Hiring managers are reviewing applications TODAY."],
        bodies=["High salary, flexible hours guaranteed.", "Send your resume and bank details to get started fast.",
                "This opportunity won't last, act now.", "Hundreds already applied, don't miss out."],
        ctas=["DM me your resume today.", "Apply via the link in my profile now.", "Message me ASAP to secure your spot."],
        emoji_pool=["🚀", "💰", "✅"], cta_prob=0.7, emoji_prob=0.5,
    )
    cap_fake_exec = cap_generic(
        openers=["As CEO, I'm proud to announce a groundbreaking initiative.", "Leading the industry into a new era.",
                 "My latest leadership insight for aspiring executives.", "Announcing a major company milestone under my leadership."],
        bodies=["True leadership means thinking bigger than everyone else.", "We're disrupting the entire sector.",
                "My journey to the top wasn't easy, but it was inevitable.", "Success comes to those who never settle."],
        ctas=["Connect with me for exclusive opportunities.", "Follow for more leadership wisdom.",
              "DM me to discuss investment opportunities."],
        emoji_pool=["💼", "🚀"], cta_prob=0.5, emoji_prob=0.2,
    )
    cap_spam_li = cap_generic(
        openers=["Grow your network 10x with this simple trick.", "I made $10k this month using this LinkedIn hack.",
                 "Everyone should be doing this in 2026.", "Stop scrolling, this could change your career."],
        bodies=["DM me and I'll show you exactly how.", "This strategy works for anyone, guaranteed.",
                "Comment 'INFO' and I'll send details.", "Limited spots in my mentorship program."],
        ctas=["Click the link in my profile.", "DM me the word 'START'.", "Book a free call today."],
        emoji_pool=["🚀", "📈", "💯"], cta_prob=0.7, emoji_prob=0.5,
    )

    li_hash_generic = ["career", "networking", "hiring", "leadership", "innovation"]
    li_hash_spam = ["hiring", "remotework", "makemoney", "opportunity", "grind"]

    def bank_li_student():
        return C.PhraseBank(
            fragments={
                "role": ["Student at {school}", "{first} | undergraduate student", "Graduate student at {school}",
                          "Aspiring professional, studying at {school}"],
                "focus": ["passionate about learning and growth", "seeking internship opportunities",
                          "active in student organizations", "combining coursework with real-world projects"],
            },
            emoji_pool=[], emoji_prob=0.05, parts_range=(2, 2),
        )

    def bank_li_dev():
        return C.PhraseBank(
            fragments={
                "role": ["Software Engineer at {company}", "Backend Developer | {city}",
                          "Full-Stack Engineer building scalable systems", "Engineer @ {company}"],
                "focus": ["passionate about clean code and mentorship", "focused on distributed systems",
                          "enjoys solving hard technical problems", "advocate for good engineering practices"],
            },
            emoji_pool=[], emoji_prob=0.05, parts_range=(2, 2),
        )

    def bank_li_biz():
        return C.PhraseBank(
            fragments={
                "role": ["{company} | Official Page", "Helping businesses grow since {year}",
                          "{city}-based company delivering results", "Innovating in our industry since {year}"],
                "focus": ["dedicated to client success", "committed to quality and innovation",
                          "proud to serve customers worldwide", "building a great place to work"],
            },
            emoji_pool=[], emoji_prob=0.02, parts_range=(2, 2),
        )

    def bank_li_consultant():
        return C.PhraseBank(
            fragments={
                "role": ["Independent Consultant | Strategy", "Management Consultant @ {company}",
                          "Helping companies navigate change", "{first} | Consultant & advisor"],
                "focus": ["specializing in operations and growth", "20+ engagements across industries",
                          "focused on measurable outcomes", "trusted advisor to leadership teams"],
            },
            emoji_pool=[], emoji_prob=0.02, parts_range=(2, 2),
        )

    def bank_li_fake_recruiter():
        return C.PhraseBank(
            fragments={
                "role": ["Senior Talent Acquisition Specialist", "Global Recruiter | Hiring Now",
                          "Connecting talent with opportunity worldwide", "Recruiter @ {company}"],
                "focus": ["hiring for multiple remote roles", "helping candidates land dream jobs fast",
                          "always looking for new talent", "high-paying opportunities available now"],
            },
            emoji_pool=["🚀"], emoji_prob=0.3, parts_range=(2, 2),
        )

    def bank_li_fake_exec():
        return C.PhraseBank(
            fragments={
                "role": ["CEO & Founder | Visionary Leader", "Chairman of {company}", "Serial Entrepreneur | CEO",
                          "President & CEO, disrupting the industry"],
                "focus": ["leading global teams to success", "building the next big thing",
                          "featured thought leader in business", "top 1% of executives worldwide"],
            },
            emoji_pool=["💼"], emoji_prob=0.3, parts_range=(2, 2),
        )

    def bank_li_spam():
        return C.PhraseBank(
            fragments={
                "role": ["Growth Hacker | LinkedIn Top Voice", "Helping you 10x your network",
                          "Digital Marketing Guru", "Passive Income Coach"],
                "focus": ["teaching thousands how to succeed online", "DM me to learn my system",
                          "self-made success story", "helping others achieve financial freedom"],
            },
            emoji_pool=["🚀", "💰"], emoji_prob=0.4, parts_range=(2, 2),
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


def linkedin_extra_builders() -> dict[str, ExtraFieldBuilder]:
    def genuine_builder(archetype_name: str) -> ExtraFieldBuilder:
        titles = JOB_TITLES_GENUINE[archetype_name]

        def build(rng, fk, n):
            job_title = [titles[int(rng.integers(0, len(titles)))] for _ in range(n)]
            company_name = [fk.company() for _ in range(n)]
            endorsements = np.round(C.sample_lognormal_clipped(rng, 0, 400, n)).astype(int)
            skills = np.round(C.sample_uniform_float(rng, 3, 50, n)).astype(int)
            return {"job_title": job_title, "company_name": company_name,
                    "endorsements_count": endorsements, "skills_count": skills}
        return build

    def fake_recruiter_builder(rng, fk, n):
        job_title = [JOB_TITLES_FAKE_RECRUITER[int(rng.integers(0, len(JOB_TITLES_FAKE_RECRUITER)))]
                     for _ in range(n)]
        company_name = [IMPLAUSIBLE_COMPANIES[int(rng.integers(0, len(IMPLAUSIBLE_COMPANIES)))]
                        if rng.random() < 0.6 else fk.company() for _ in range(n)]
        endorsements = np.round(C.sample_lognormal_clipped(rng, 0, 20, n)).astype(int)
        skills = np.round(C.sample_uniform_float(rng, 0, 8, n)).astype(int)
        return {"job_title": job_title, "company_name": company_name,
                "endorsements_count": endorsements, "skills_count": skills}

    def fake_exec_builder(rng, fk, n):
        job_title = [JOB_TITLES_FAKE_EXEC[int(rng.integers(0, len(JOB_TITLES_FAKE_EXEC)))] for _ in range(n)]
        company_name = [IMPLAUSIBLE_COMPANIES[int(rng.integers(0, len(IMPLAUSIBLE_COMPANIES)))]
                        for _ in range(n)]
        endorsements = np.round(C.sample_lognormal_clipped(rng, 0, 15, n)).astype(int)
        skills = np.round(C.sample_uniform_float(rng, 0, 6, n)).astype(int)
        return {"job_title": job_title, "company_name": company_name,
                "endorsements_count": endorsements, "skills_count": skills}

    def spam_builder(rng, fk, n):
        job_title = [JOB_TITLES_SPAM[int(rng.integers(0, len(JOB_TITLES_SPAM)))] for _ in range(n)]
        company_name = ["Self-Employed" if rng.random() < 0.7 else fk.company() for _ in range(n)]
        endorsements = np.round(C.sample_lognormal_clipped(rng, 0, 10, n)).astype(int)
        skills = np.round(C.sample_uniform_float(rng, 0, 5, n)).astype(int)
        return {"job_title": job_title, "company_name": company_name,
                "endorsements_count": endorsements, "skills_count": skills}

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

    def make(name):
        lo, hi = ranges[name]

        def build(rng, fk, n):
            mutual = np.round(C.sample_lognormal_clipped(rng, lo, hi, n)).astype(int)
            return {"mutual_friends_count": mutual}
        return build

    return {name: make(name) for name in ranges}


# ===========================================================================
# Orchestration
# ===========================================================================

PLATFORM_SPECS = {
    "ig": dict(
        platform="instagram", prefix="ig", archetypes_fn=instagram_archetypes,
        posts_count_field="posts_count", primary_field="followers_count", secondary_field="following_count",
        engagement_denominator_fields=["followers_count"], extra_field_builders={}, text_field="bio",
        profile_column_order=["user_id", "username", "bio", "followers_count", "following_count", "posts_count",
                               "account_age_days", "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                               "is_verified", "has_website", "has_location", "profile_completion_score",
                               "archetype", "is_fake"],
    ),
    "tw": dict(
        platform="twitter", prefix="tw", archetypes_fn=twitter_archetypes,
        posts_count_field="tweets_count", primary_field="followers_count", secondary_field="following_count",
        engagement_denominator_fields=["followers_count"], extra_field_builders={}, text_field="bio",
        profile_column_order=["user_id", "username", "bio", "followers_count", "following_count", "tweets_count",
                               "account_age_days", "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                               "is_verified", "has_website", "has_location", "profile_completion_score",
                               "archetype", "is_fake"],
    ),
    "fb": dict(
        platform="facebook", prefix="fb", archetypes_fn=facebook_archetypes,
        posts_count_field="posts_count", primary_field="friends_count", secondary_field="followers_count",
        engagement_denominator_fields=["friends_count", "followers_count"],
        extra_field_builders=facebook_extra_builders(), text_field="bio",
        profile_column_order=["user_id", "username", "bio", "friends_count", "followers_count", "posts_count",
                               "account_age_days", "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                               "mutual_friends_count", "is_verified", "has_website", "has_location",
                               "profile_completion_score", "archetype", "is_fake"],
    ),
    "li": dict(
        platform="linkedin", prefix="li", archetypes_fn=linkedin_archetypes,
        posts_count_field="posts_count", primary_field="connections_count", secondary_field=None,
        engagement_denominator_fields=["connections_count"],
        extra_field_builders=linkedin_extra_builders(), text_field="headline",
        profile_column_order=["user_id", "username", "headline", "connections_count", "posts_count",
                               "account_age_days", "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                               "job_title", "company_name", "endorsements_count", "skills_count",
                               "is_verified", "has_website", "has_location", "profile_completion_score",
                               "archetype", "is_fake"],
    ),
}

PROFILE_SCHEMA = {
    "ig": {"user_id": "str", "username": "str", "bio": "str", "followers_count": "int", "following_count": "int",
           "posts_count": "int", "account_age_days": "int", "avg_likes": "float", "avg_comments": "float",
           "avg_shares": "float", "engagement_rate": "float", "is_verified": "bool", "has_website": "bool",
           "has_location": "bool", "profile_completion_score": "float", "archetype": "str", "is_fake": "int"},
    "tw": {"user_id": "str", "username": "str", "bio": "str", "followers_count": "int", "following_count": "int",
           "tweets_count": "int", "account_age_days": "int", "avg_likes": "float", "avg_comments": "float",
           "avg_shares": "float", "engagement_rate": "float", "is_verified": "bool", "has_website": "bool",
           "has_location": "bool", "profile_completion_score": "float", "archetype": "str", "is_fake": "int"},
    "fb": {"user_id": "str", "username": "str", "bio": "str", "friends_count": "int", "followers_count": "int",
           "posts_count": "int", "account_age_days": "int", "avg_likes": "float", "avg_comments": "float",
           "avg_shares": "float", "engagement_rate": "float", "mutual_friends_count": "int", "is_verified": "bool",
           "has_website": "bool", "has_location": "bool", "profile_completion_score": "float", "archetype": "str",
           "is_fake": "int"},
    "li": {"user_id": "str", "username": "str", "headline": "str", "connections_count": "int", "posts_count": "int",
           "account_age_days": "int", "avg_likes": "float", "avg_comments": "float", "avg_shares": "float",
           "engagement_rate": "float", "job_title": "str", "company_name": "str", "endorsements_count": "int",
           "skills_count": "int", "is_verified": "bool", "has_website": "bool", "has_location": "bool",
           "profile_completion_score": "float", "archetype": "str", "is_fake": "int"},
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
        text_field=spec["text_field"],
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
