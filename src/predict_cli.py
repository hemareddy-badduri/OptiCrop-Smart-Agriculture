"""Command-line crop predictor using the saved ``models/model.pkl`` bundle.

Examples
--------
Predict from explicit values:
    python src/predict_cli.py --N 80 --P 48 --K 40 --temperature 23 \
        --humidity 82 --ph 6.4 --rainfall 225

Predict from a JSON file (handy for batch testing):
    python src/predict_cli.py --json sample_input.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.preprocess import transform_prediction_features  # noqa: E402


def load_bundle():
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found at {config.MODEL_FILE}. "
            "Run `python src/train_models.py` first."
        )
    with open(config.MODEL_FILE, "rb") as f:
        return pickle.load(f)


def predict(features: dict) -> str:
    bundle = load_bundle()
    cols = bundle["feature_columns"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    row = [transform_prediction_features(features, cols)]
    scaled = scaler.transform(row)
    pred = model.predict(scaled)[0]
    return str(pred)


def main() -> None:
    p = argparse.ArgumentParser(description="Predict the best crop from input features.")
    p.add_argument("--json", type=str, help="path to a JSON file with feature values")
    for col in config.FEATURE_COLUMNS:
        p.add_argument(f"--{col}", type=float, required=False)
    args = p.parse_args()

    if args.json:
        with open(args.json) as f:
            features = json.load(f)
    else:
        features = {}
        for col in config.FEATURE_COLUMNS:
            val = getattr(args, col)
            if val is None:
                p.error(f"--{col} is required (or supply --json FILE)")
            features[col] = val

    crop = predict(features)
    print(f"Recommended crop: {crop.upper()}")
    print("Based on inputs:")
    for k, v in features.items():
        print(f"  {k:12s}: {v}")


if __name__ == "__main__":
    main()
