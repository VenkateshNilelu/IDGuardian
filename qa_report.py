"""
IDGuardian dataset QA pass -- deeper checks than generation-time validation.

Runs, per platform:
  1. Structural correctness (orphans, label consistency, aggregate consistency,
     volume/balance), reporting violation counts rather than pass/fail.
  2. Text duplication rates per archetype for bios/headlines and captions,
     plus random text samples for manual spot-checking.
  3. Numeric distribution sanity (min/max/mean/std per archetype) with flags
     for impossible values and for fake-vs-genuine skew that isn't showing up.
  4. Per-feature leakage check: a single-feature classifier per numeric column.
  5. Overall separability: a full baseline classifier (logistic regression +
     shallow XGBoost) on all behavioral numeric features.

Writes data/qa_report.md and data/text_samples.md. Also prints a condensed
summary to stdout. This is read-only with respect to the CSVs -- it never
edits them; tuning happens by changing generate_dataset.py/common.py and
regenerating.

Usage: python qa_report.py
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
QA_SEED = 42
TARGET_PROFILES = 5000
TARGET_POSTS = 50000
DUP_FLAG_THRESHOLD = 5.0       # percent
LEAKAGE_FLAG_THRESHOLD = 0.98  # accuracy
LINKEDIN_CONNECTIONS_CAP = 30000


@dataclass
class PlatformConfig:
    key: str
    posts_count_col: str
    text_col: str
    numeric_cols: list          # raw numeric/bool columns present in the profiles CSV
    bool_cols: list
    derived: dict = field(default_factory=dict)   # name -> callable(df) -> Series
    count_caps: dict = field(default_factory=dict)  # column -> max plausible value


PLATFORM_CONFIGS = {
    "instagram": PlatformConfig(
        key="instagram", posts_count_col="posts_count", text_col="bio",
        numeric_cols=["followers_count", "following_count", "posts_count", "account_age_days",
                      "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                      "profile_completion_score"],
        bool_cols=["is_verified", "has_website", "has_location"],
        derived={
            "follow_ratio": lambda df: df["followers_count"] / (df["following_count"] + 1),
            "follower_gap": lambda df: df["followers_count"] - df["following_count"],
        },
    ),
    "twitter": PlatformConfig(
        key="twitter", posts_count_col="tweets_count", text_col="bio",
        numeric_cols=["followers_count", "following_count", "tweets_count", "account_age_days",
                      "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                      "profile_completion_score"],
        bool_cols=["is_verified", "has_website", "has_location"],
        derived={
            "follow_ratio": lambda df: df["followers_count"] / (df["following_count"] + 1),
            "follower_gap": lambda df: df["followers_count"] - df["following_count"],
        },
    ),
    "facebook": PlatformConfig(
        key="facebook", posts_count_col="posts_count", text_col="bio",
        numeric_cols=["friends_count", "followers_count", "posts_count", "account_age_days",
                      "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                      "mutual_friends_count", "profile_completion_score"],
        bool_cols=["is_verified", "has_website", "has_location"],
        derived={
            "friend_follower_ratio": lambda df: df["friends_count"] / (df["followers_count"] + 1),
            "mutual_friend_ratio": lambda df: df["mutual_friends_count"] / (df["friends_count"] + 1),
        },
    ),
    "linkedin": PlatformConfig(
        key="linkedin", posts_count_col="posts_count", text_col="headline",
        numeric_cols=["connections_count", "posts_count", "account_age_days",
                      "avg_likes", "avg_comments", "avg_shares", "engagement_rate",
                      "endorsements_count", "skills_count", "profile_completion_score"],
        bool_cols=["is_verified", "has_website", "has_location"],
        derived={
            "endorsements_per_skill": lambda df: df["endorsements_count"] / (df["skills_count"] + 1),
        },
        count_caps={"connections_count": LINKEDIN_CONNECTIONS_CAP},
    ),
}


def load_platform(cfg: PlatformConfig):
    profiles = pd.read_csv(os.path.join(DATA_DIR, f"{cfg.key}_profiles.csv"))
    posts = pd.read_csv(os.path.join(DATA_DIR, f"{cfg.key}_posts.csv"), keep_default_na=False)
    return profiles, posts


# ---------------------------------------------------------------------------
# 1. Structural correctness
# ---------------------------------------------------------------------------

def structural_checks(cfg: PlatformConfig, profiles: pd.DataFrame, posts: pd.DataFrame, tol=1e-3) -> dict:
    out = {}
    profile_ids = set(profiles["user_id"])
    orphan_mask = ~posts["user_id"].isin(profile_ids)
    out["n_orphan_posts"] = int(orphan_mask.sum())

    merged = posts.merge(profiles[["user_id", "archetype", "is_fake"]], on="user_id",
                          how="left", suffixes=("_post", "_profile"))
    out["n_archetype_mismatch"] = int((merged["archetype_post"] != merged["archetype_profile"]).sum())
    out["n_is_fake_mismatch"] = int((merged["is_fake_post"] != merged["is_fake_profile"]).sum())

    agg = posts.groupby("user_id").agg(
        c_avg_likes=("likes", "mean"), c_avg_comments=("comments", "mean"),
        c_avg_shares=("shares", "mean"), c_count=("likes", "size"),
    )
    chk = profiles.merge(agg, on="user_id", how="left")
    out["n_avg_likes_mismatch"] = int((chk["avg_likes"] - chk["c_avg_likes"]).abs().gt(tol).sum())
    out["n_avg_comments_mismatch"] = int((chk["avg_comments"] - chk["c_avg_comments"]).abs().gt(tol).sum())
    out["n_avg_shares_mismatch"] = int((chk["avg_shares"] - chk["c_avg_shares"]).abs().gt(tol).sum())
    out["n_count_mismatch"] = int((chk[cfg.posts_count_col] != chk["c_count"]).sum())

    out["n_profiles"] = len(profiles)
    out["n_posts"] = len(posts)
    out["profiles_target_diff"] = len(profiles) - TARGET_PROFILES
    out["posts_target_diff_pct"] = round(100 * (len(posts) - TARGET_POSTS) / TARGET_POSTS, 2)

    fake_pct = 100 * profiles["is_fake"].mean()
    out["overall_fake_pct"] = round(fake_pct, 2)
    out["overall_genuine_pct"] = round(100 - fake_pct, 2)

    per_arch = profiles.groupby("archetype").agg(n=("user_id", "size"), is_fake=("is_fake", "first"))
    out["per_archetype_counts"] = per_arch.to_dict(orient="index")

    return out


# ---------------------------------------------------------------------------
# 2. Text quality / diversity
# ---------------------------------------------------------------------------

def duplicate_pct(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    vc = series.value_counts()
    dup_rows = int(vc[vc > 1].sum())
    return round(100 * dup_rows / len(series), 2)


def text_checks(cfg: PlatformConfig, profiles: pd.DataFrame, posts: pd.DataFrame) -> dict:
    out = {"bio_dup_pct_by_archetype": {}, "caption_dup_pct_by_archetype": {}}
    for arch, grp in profiles.groupby("archetype"):
        out["bio_dup_pct_by_archetype"][arch] = duplicate_pct(grp[cfg.text_col])
    for arch, grp in posts.groupby("archetype"):
        out["caption_dup_pct_by_archetype"][arch] = duplicate_pct(grp["caption"])
    return out


def write_text_samples(fh, platform_name: str, cfg: PlatformConfig, profiles: pd.DataFrame,
                        posts: pd.DataFrame, rng: random.Random):
    fh.write(f"\n## {platform_name.capitalize()}\n")
    for arch, grp in profiles.groupby("archetype"):
        fh.write(f"\n### {arch}\n\n**Sample {cfg.text_col}s:**\n\n")
        sample = grp[cfg.text_col].tolist()
        picks = rng.sample(sample, min(5, len(sample)))
        for s in picks:
            fh.write(f"- {s}\n")
        post_grp = posts[posts["archetype"] == arch]
        fh.write("\n**Sample captions:**\n\n")
        csample = post_grp["caption"].tolist()
        cpicks = rng.sample(csample, min(5, len(csample)))
        for s in cpicks:
            fh.write(f"- {s}\n")


# ---------------------------------------------------------------------------
# 3. Numeric distribution sanity
# ---------------------------------------------------------------------------

def with_derived(cfg: PlatformConfig, profiles: pd.DataFrame) -> pd.DataFrame:
    df = profiles.copy()
    for name, fn in cfg.derived.items():
        df[name] = fn(df)
    for c in cfg.bool_cols:
        df[c] = df[c].astype(int)
    return df


def numeric_checks(cfg: PlatformConfig, profiles: pd.DataFrame) -> dict:
    df = with_derived(cfg, profiles)
    all_numeric = cfg.numeric_cols + list(cfg.derived.keys()) + cfg.bool_cols
    out = {"describe_by_archetype": {}, "flags": []}

    for col in cfg.numeric_cols + list(cfg.derived.keys()):
        neg = df[df[col] < 0]
        if len(neg) > 0 and col not in ("follower_gap",):
            out["flags"].append(f"{col}: {len(neg)} negative values (impossible for a count/rate column)")

    if "engagement_rate" in df.columns:
        bad = df[(df["engagement_rate"] < 0) | (df["engagement_rate"] > 2.0)]
        if len(bad) > 0:
            out["flags"].append(f"engagement_rate: {len(bad)} rows outside sane [0, 2.0] range "
                                 f"(max={df['engagement_rate'].max():.4f})")

    if "profile_completion_score" in df.columns:
        bad = df[(df["profile_completion_score"] < 0) | (df["profile_completion_score"] > 1)]
        if len(bad) > 0:
            out["flags"].append(f"profile_completion_score: {len(bad)} rows outside [0,1]")

    for col, cap in cfg.count_caps.items():
        bad = df[df[col] > cap]
        if len(bad) > 0:
            out["flags"].append(f"{col}: {len(bad)} rows exceed platform cap of {cap}")

    for col in all_numeric:
        stats = df.groupby("archetype")[col].agg(["min", "max", "mean", "std"]).round(4)
        out["describe_by_archetype"][col] = stats.to_dict(orient="index")

    # fake-vs-genuine skew check: for key signal columns, compare genuine vs fake means
    # and flag if they're statistically indistinguishable (Cohen's d ~ 0)
    skew_cols = [c for c in ["account_age_days", "engagement_rate", "follow_ratio",
                              "friend_follower_ratio", "mutual_friend_ratio"] if c in df.columns]
    out["skew_check"] = {}
    for col in skew_cols:
        genuine = df.loc[df["is_fake"] == 0, col]
        fake = df.loc[df["is_fake"] == 1, col]
        pooled_std = np.sqrt((genuine.std() ** 2 + fake.std() ** 2) / 2)
        cohens_d = 0.0 if pooled_std == 0 else abs(genuine.mean() - fake.mean()) / pooled_std
        out["skew_check"][col] = {
            "genuine_mean": round(float(genuine.mean()), 4),
            "fake_mean": round(float(fake.mean()), 4),
            "cohens_d": round(float(cohens_d), 3),
        }
        if cohens_d < 0.2:
            out["flags"].append(f"{col}: genuine vs fake means look statistically indistinguishable "
                                 f"(Cohen's d={cohens_d:.3f} < 0.2) -- expected signal is missing")

    return out


# ---------------------------------------------------------------------------
# 4. Per-feature leakage check
# ---------------------------------------------------------------------------

def leakage_check(cfg: PlatformConfig, profiles: pd.DataFrame) -> dict:
    df = with_derived(cfg, profiles)
    all_numeric = cfg.numeric_cols + list(cfg.derived.keys()) + cfg.bool_cols
    y = df["is_fake"].to_numpy()
    out = {}
    for col in all_numeric:
        X = df[[col]].to_numpy(dtype=float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=QA_SEED, stratify=y)
        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(scaler.transform(X_train), y_train)
        pred = clf.predict(scaler.transform(X_test))
        proba = clf.predict_proba(scaler.transform(X_test))[:, 1]
        acc = accuracy_score(y_test, pred)
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")
        out[col] = {"accuracy": round(float(acc), 4), "auc": round(float(auc), 4)}
    return out


# ---------------------------------------------------------------------------
# 5. Overall separability
# ---------------------------------------------------------------------------

def separability_check(cfg: PlatformConfig, profiles: pd.DataFrame) -> dict:
    df = with_derived(cfg, profiles)
    feature_cols = cfg.numeric_cols + list(cfg.derived.keys()) + cfg.bool_cols
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["is_fake"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=QA_SEED, stratify=y)

    results = {}

    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(scaler.transform(X_train), y_train)
    pred = lr.predict(scaler.transform(X_test))
    proba = lr.predict_proba(scaler.transform(X_test))[:, 1]
    results["logistic_regression"] = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }

    xgb_clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=QA_SEED,
        eval_metric="logloss",
    )
    xgb_clf.fit(X_train, y_train)
    pred = xgb_clf.predict(X_test)
    proba = xgb_clf.predict_proba(X_test)[:, 1]
    results["xgboost"] = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all() -> dict:
    results = {}
    for name, cfg in PLATFORM_CONFIGS.items():
        profiles, posts = load_platform(cfg)
        results[name] = {
            "structural": structural_checks(cfg, profiles, posts),
            "text": text_checks(cfg, profiles, posts),
            "numeric": numeric_checks(cfg, profiles),
            "leakage": leakage_check(cfg, profiles),
            "separability": separability_check(cfg, profiles),
        }
    return results


def print_summary(results: dict):
    for platform, r in results.items():
        print(f"\n=== {platform.upper()} ===")
        s = r["structural"]
        print(f"  profiles={s['n_profiles']} posts={s['n_posts']} "
              f"balance={s['overall_genuine_pct']}/{s['overall_fake_pct']}")
        print(f"  orphans={s['n_orphan_posts']} archetype_mismatch={s['n_archetype_mismatch']} "
              f"is_fake_mismatch={s['n_is_fake_mismatch']} agg_mismatch="
              f"{s['n_avg_likes_mismatch']}/{s['n_avg_comments_mismatch']}/"
              f"{s['n_avg_shares_mismatch']}/{s['n_count_mismatch']}")

        bio_flags = {k: v for k, v in r["text"]["bio_dup_pct_by_archetype"].items() if v > DUP_FLAG_THRESHOLD}
        cap_flags = {k: v for k, v in r["text"]["caption_dup_pct_by_archetype"].items() if v > DUP_FLAG_THRESHOLD}
        if bio_flags:
            print(f"  [FLAG] bio/headline duplicate% > {DUP_FLAG_THRESHOLD}: {bio_flags}")
        if cap_flags:
            print(f"  [FLAG] caption duplicate% > {DUP_FLAG_THRESHOLD}: {cap_flags}")

        for flag in r["numeric"]["flags"]:
            print(f"  [FLAG] {flag}")

        leak_flags = {k: v for k, v in r["leakage"].items() if v["accuracy"] > LEAKAGE_FLAG_THRESHOLD}
        if leak_flags:
            print(f"  [FLAG] single-feature leakage > {LEAKAGE_FLAG_THRESHOLD}: {leak_flags}")

        print(f"  separability: LR acc={r['separability']['logistic_regression']['accuracy']} "
              f"auc={r['separability']['logistic_regression']['roc_auc']} | "
              f"XGB acc={r['separability']['xgboost']['accuracy']} "
              f"auc={r['separability']['xgboost']['roc_auc']}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    results = run_all()
    print_summary(results)

    rng = random.Random(QA_SEED)
    with open(os.path.join(DATA_DIR, "text_samples.md"), "w", encoding="utf-8") as fh:
        fh.write("# IDGuardian Text Samples (for manual spot-check)\n")
        for name, cfg in PLATFORM_CONFIGS.items():
            profiles, posts = load_platform(cfg)
            write_text_samples(fh, name, cfg, profiles, posts, rng)
    print(f"\nWrote {os.path.join(DATA_DIR, 'text_samples.md')}")
    return results


if __name__ == "__main__":
    main()
