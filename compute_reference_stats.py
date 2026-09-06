"""
Precompute per-platform, per-feature "typical genuine" / "typical fake" reference
values from the main training dataset, for the web app's behavioral-layer
explanations (e.g. "account_age_days=25 vs. genuine-typical ~1900").

Saved to models/behavioral_reference_stats.json -- a small artifact checked into
git -- so app.py never needs the full 5,000-row training CSVs at runtime (those
are gitignored/regenerable; a fresh clone won't have them until someone runs
generate_dataset.py).

Usage: python compute_reference_stats.py   (run once after training/retraining)
"""
import json
import os

import train_models as T

MODELS_DIR = T.MODELS_DIR


def main():
    stats = {}
    for p in T.PLATFORMS:
        df = T.load_platform(p)
        cols = T.behavioral_columns(p)
        genuine = df[df["is_fake"] == 0]
        fake = df[df["is_fake"] == 1]
        stats[p] = {
            c: {
                "genuine_mean": round(float(genuine[c].mean()), 4),
                "fake_mean": round(float(fake[c].mean()), 4),
            }
            for c in cols
        }

    out_path = os.path.join(MODELS_DIR, "behavioral_reference_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
