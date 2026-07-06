"""Data preprocessing for OptiCrop (Epic 3).

Responsibilities
----------------
1. Check / handle null values
2. Detect & treat outliers with the IQR rule (log-transform on Potassium)
3. Extract seasonal crop groupings (summer / winter / rainy)
4. Split the data into train / test sets

The functions here are reused by `train_models.py` so that the exact same
pipeline used during EDA is used for training.

Run as a script to print a preprocessing report:

    python src/preprocess.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Seasonal crop mapping (which crops suit which season)
# ---------------------------------------------------------------------------
SEASONAL_CROPS = {
    "summer": ["muskmelon", "watermelon", "pigeonpeas", "blackgram", "mango",
               "papaya", "maize", "mothbeans"],
    "winter": ["chickpea", "lentil", "wheat", "mustard", "orange", "apple",
               "grapes"],
    "rainy": ["rice", "jute", "cotton", "coconut", "coffee", "pomegranate",
              "banana", "mungbean", "kidneybeans"],
}


def load_data() -> pd.DataFrame:
    if not config.DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.DATA_FILE}. "
            "Run `python data/generate_synthetic_data.py` first."
        )
    return pd.read_csv(config.DATA_FILE)


# ---------------------------------------------------------------------------
# 1. Null handling
# ---------------------------------------------------------------------------
def check_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Report nulls and drop any rows containing them (none expected)."""
    nulls = df.isnull().sum()
    print("[Nulls] Per-column null counts:")
    print(nulls.to_string())
    if nulls.sum() > 0:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        print(f"[Nulls] Dropped {before - len(df)} rows with missing values.")
    else:
        print("[Nulls] No missing values found - data is clean.")
    return df


# ---------------------------------------------------------------------------
# 2. Outlier detection (IQR) + log-transform on Potassium
# ---------------------------------------------------------------------------
def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def handle_outliers(df: pd.DataFrame, plot: bool = False) -> pd.DataFrame:
    """Identify outliers via the IQR rule, then log-transform Potassium (K).

    K has a heavy right-skew in the real dataset (a few crops like apple and
    grapes need very high K). A log1p transform normalises it without deleting
    those legitimate high-K samples.
    """
    df = df.copy()
    print("\n[Outliers] IQR bounds per feature:")
    for col in config.FEATURE_COLUMNS:
        lo, hi = iqr_bounds(df[col])
        out = ((df[col] < lo) | (df[col] > hi)).sum()
        print(f"  {col:12s}: bounds=({lo:7.2f}, {hi:7.2f})  outliers={out}")

    df["K"] = np.log1p(df["K"])
    print("[Outliers] Applied log1p transform on 'K' to reduce skew.")
    return df


def transform_prediction_features(features: dict, columns: list[str] | None = None) -> list[float]:
    """Apply training-time feature transforms to one prediction input row.

    Form and CLI users enter the original agronomic values. Training applies a
    log1p transform to Potassium (K), so inference must do the same before the
    saved scaler/model see the row.
    """
    columns = columns or config.FEATURE_COLUMNS
    transformed = {col: float(features[col]) for col in config.FEATURE_COLUMNS}
    if transformed["K"] < 0:
        raise ValueError("Potassium (K) must be zero or greater.")
    transformed["K"] = np.log1p(transformed["K"])
    return [transformed[col] for col in columns]


# ---------------------------------------------------------------------------
# 3. Seasonal crop extraction
# ---------------------------------------------------------------------------
def seasonal_summary(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return crops present in the data grouped by season."""
    present = set(df[config.TARGET_COLUMN].unique())
    summary = {}
    for season, crops in SEASONAL_CROPS.items():
        summary[season] = sorted(c for c in crops if c in present)
        print(f"[Season] {season:7s}: {', '.join(summary[season])}")
    return summary


# ---------------------------------------------------------------------------
# 4. Train/test split
# ---------------------------------------------------------------------------
def make_split(df: pd.DataFrame):
    X = df[config.FEATURE_COLUMNS].values
    y = df[config.TARGET_COLUMN].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"\n[Split] Train: {X_tr.shape}, Test: {X_te.shape}")
    return X_tr, X_te, y_tr, y_te


def build_pipeline(plot: bool = False) -> tuple:
    """Run the full preprocessing pipeline and return train/test arrays."""
    df = load_data()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")
    df = check_nulls(df)
    df = handle_outliers(df, plot=plot)
    print()
    seasonal_summary(df)
    return make_split(df)


def main() -> None:
    X_tr, X_te, y_tr, y_te = build_pipeline(plot=False)
    print("\n[OK] Preprocessing complete.")
    print(f"  X_train: {X_tr.shape}  y_train classes: {len(np.unique(y_tr))}")


if __name__ == "__main__":
    main()
