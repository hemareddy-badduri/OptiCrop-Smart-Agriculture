"""Central configuration for the OptiCrop project.

All paths and constants are defined here so every script (EDA, preprocessing,
training, web app) resolves the same locations regardless of where it is run
from. Edit values here to reconfigure the project.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
WEB_DIR = ROOT_DIR / "web"

# Primary dataset (real Kaggle schema). Drop the real file here to override
# the synthetic one — no code changes required.
DATA_FILE = DATA_DIR / "Crop_recommendation.csv"

# Trained model artifact produced by src/train_models.py
MODEL_FILE = MODEL_DIR / "model.pkl"

# ---------------------------------------------------------------------------
# Feature schema (matches the real Kaggle dataset exactly)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
