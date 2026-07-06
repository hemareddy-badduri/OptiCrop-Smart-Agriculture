"""Generate a realistic synthetic ``Crop_recommendation.csv`` dataset.

The real Kaggle dataset ("Crop recommendation") has 7 features
(N, P, K, temperature, humidity, ph, rainfall) and 22 crop labels, with 100
samples per crop (2200 rows). This generator reproduces that *exact schema* and
uses per-crop mean/standard-deviation profiles derived from the public dataset
so that any model trained here behaves realistically. When you obtain the real
CSV from Kaggle, simply replace ``data/Crop_recommendation.csv`` — every script
in the project will pick it up unchanged.

Usage:
    python data/generate_synthetic_data.py            # default 100 rows/crop
    python data/generate_synthetic_data.py --rows 200 # 200 rows/crop
"""
import argparse
import sys

import numpy as np
import pandas as pd

# Resolve project config regardless of CWD
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Per-crop parameter profiles
# Columns: N, P, K, temperature, humidity, ph, rainfall  (mean values)
# Standard deviations are derived below as a fraction of the mean/range so the
# generated spread is realistic but classes remain separable.
# These means mirror the well-known public "Crop recommendation" dataset.
# ---------------------------------------------------------------------------
CROP_PROFILES = {
    # crop:           [N,   P,   K,   temp, hum, ph,  rain]
    "rice":         [80,  48,  40,  23,   82,  6.4, 225],
    "maize":        [78,  48,  20,  22,   64,  6.3, 95],
    "chickpea":     [40,  68,  80,  18,   17,  7.3, 80],
    "kidneybeans":  [21,  67,  20,  20,   22,  5.7, 105],
    "pigeonpeas":   [21,  68,  20,  27,   48,  5.8, 150],
    "mothbeans":    [21,  67,  20,  27,   58,  6.8, 50],
    "mungbean":     [21,  47,  20,  28,   60,  6.9, 48],
    "blackgram":    [40,  67,  20,  30,   65,  7.1, 70],
    "lentil":       [18,  68,  19,  24,   65,  6.9, 46],
    "pomegranate":  [19,  17,  40,  21,   90,  6.4, 108],
    "banana":       [100, 82,  50,  27,   80,  6.0, 105],
    "mango":        [20,  26,  30,  31,   50,  5.8, 95],
    "grapes":       [23,  132, 200, 23,   82,  6.0, 70],
    "watermelon":   [85,  25,  50,  26,   50,  6.5, 55],
    "muskmelon":    [100, 25,  50,  29,   90,  6.5, 25],
    "apple":        [21,  134, 200, 22,   92,  5.9, 115],
    "orange":       [20,  17,  10,  22,   92,  7.0, 110],
    "papaya":       [50,  59,  50,  34,   90,  6.7, 145],
    "coconut":      [22,  17,  31,  27,   95,  5.9, 175],
    "cotton":       [118, 46,  20,  24,   80,  6.9, 80],
    "jute":         [79,  47,  40,  25,   81,  6.7, 175],
    "coffee":       [101, 29,  30,  25,   58,  6.8, 158],
}

# Relative noise per feature (fraction of mean). ph gets a small absolute noise.
NOISE = {
    "N": 0.12, "P": 0.12, "K": 0.12,
    "temperature": 0.06, "humidity": 0.06,
    "ph": 0.08, "rainfall": 0.12,
}

# Hard lower/upper clipping bounds for physical plausibility
BOUNDS = {
    "N": (0, 150),
    "P": (0, 150),
    "K": (0, 210),
    "temperature": (8, 45),
    "humidity": (10, 100),
    "ph": (3.5, 10.0),
    "rainfall": (10, 300),
}

COLUMNS = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]


def _sample_value(col: str, mean: float, rng: np.random.Generator) -> float:
    """Draw a single noisy, clipped value for one feature."""
    scale = mean * NOISE[col] if mean != 0 else NOISE[col]
    # ph/humidity handled as small absolute spread
    if col == "ph":
        scale = max(0.25, scale)
    val = rng.normal(mean, scale)
    lo, hi = BOUNDS[col]
    return float(np.clip(val, lo, hi))


def generate(rows_per_crop: int = 100, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Generate the full dataset DataFrame."""
    rng = np.random.default_rng(seed)
    records = []
    for crop, means in CROP_PROFILES.items():
        prof = dict(zip(config.FEATURE_COLUMNS, means))
        for _ in range(rows_per_crop):
            row = {col: _sample_value(col, prof[col], rng) for col in config.FEATURE_COLUMNS}
            # Round numerics to realistic precision
            row["N"] = round(row["N"])
            row["P"] = round(row["P"])
            row["K"] = round(row["K"])
            row["temperature"] = round(row["temperature"], 2)
            row["humidity"] = round(row["humidity"], 2)
            row["ph"] = round(row["ph"], 2)
            row["rainfall"] = round(row["rainfall"], 2)
            row[config.TARGET_COLUMN] = crop
            records.append(row)
    df = pd.DataFrame(records, columns=COLUMNS)
    # Shuffle so train/test split isn't grouped by class
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic crop recommendation data.")
    parser.add_argument("--rows", type=int, default=100, help="rows per crop (default: 100)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE, help="random seed")
    parser.add_argument("--out", type=str, default=str(config.DATA_FILE), help="output CSV path")
    args = parser.parse_args()

    df = generate(rows_per_crop=args.rows, seed=args.seed)
    out_path = __import__("pathlib").Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Generated {len(df)} rows ({args.rows}/crop x {df['label'].nunique()} crops)")
    print(f"[OK] Saved to: {out_path}")
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())
    print("\nDescriptive statistics:")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
