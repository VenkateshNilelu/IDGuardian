"""
Generate a small, disjoint "demo" dataset for the IDGuardian web app's
username-lookup feature.

This is NOT training data -- it exists so a user can type a username into the
app and see a real (synthetic) profile scored end-to-end, without the app
scraping any live platform. Each platform gets its own RNG/Faker salt
("<platform>_demo" instead of "<platform>") and its own user_id prefix
("igdemo_" instead of "ig_"), so these profiles are guaranteed to never
overlap with the 5,000/platform dataset the models were actually trained and
tested on -- otherwise a lookup could land on a profile the model memorized
during training, which would make the demo dishonestly perfect rather than a
real test of generalization.

Usage: python generate_demo_dataset.py
"""
import os

import common as C
from generate_dataset import (
    DATA_DIR, NULLABLE_COLS, OVERLAP_FRAC, POST_SCHEMA, PROFILE_SCHEMA,
    PLATFORM_SPECS, TEXT_OVERLAP_FRAC, build_platform_dataset,
)

DEMO_N_PROFILES = 200


def run_demo_platform(key: str):
    spec = PLATFORM_SPECS[key]
    archetypes = spec["archetypes_fn"]()
    profiles, posts = build_platform_dataset(
        platform=spec["platform"] + "_demo",
        prefix=spec["prefix"] + "demo",
        archetypes=archetypes,
        n_profiles=DEMO_N_PROFILES,
        posts_count_field=spec["posts_count_field"],
        primary_field=spec["primary_field"],
        secondary_field=spec["secondary_field"],
        engagement_denominator_fields=spec["engagement_denominator_fields"],
        extra_field_builders=spec["extra_field_builders"],
        profile_column_order=spec["profile_column_order"],
        text_field=spec["text_field"],
        overlap_frac=OVERLAP_FRAC,
        text_overlap_frac=TEXT_OVERLAP_FRAC,
    )
    os.makedirs(DATA_DIR, exist_ok=True)
    profiles.to_csv(os.path.join(DATA_DIR, f"demo_{spec['platform']}_profiles.csv"), index=False)
    posts.to_csv(os.path.join(DATA_DIR, f"demo_{spec['platform']}_posts.csv"), index=False)

    report, ok = C.validate_platform(
        spec["platform"] + "_demo", profiles, posts, PROFILE_SCHEMA[key], POST_SCHEMA,
        nullable_cols=NULLABLE_COLS,
    )
    return profiles, posts, report, ok


def main():
    all_ok = True
    all_lines: list[str] = []
    for key in ["ig", "fb", "li", "tw"]:
        profiles, posts, report, ok = run_demo_platform(key)
        all_ok = all_ok and ok
        all_lines.extend(report)
        print(f"{PLATFORM_SPECS[key]['platform']} demo: profiles={len(profiles)} "
              f"posts={len(posts)} validation={'OK' if ok else 'FAILED'}")

    with open(os.path.join(DATA_DIR, "demo_validation_report.txt"), "w", encoding="utf-8") as f:
        f.write("IDGuardian DEMO dataset validation report\n")
        f.write(f"Overall: {'ALL PLATFORMS PASSED' if all_ok else 'SOME CHECKS FAILED'}\n\n")
        f.write("\n".join(all_lines))

    print(f"\nOverall: {'ALL PLATFORMS PASSED' if all_ok else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
