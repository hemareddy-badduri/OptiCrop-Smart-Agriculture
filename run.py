"""OptiCrop application entry point.

Launches the Flask web server. Run from the project root:

    python run.py

Environment variables:
    PORT  — port to bind (default 5000)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

# Warn (but still start) if the model isn't trained yet
if not config.MODEL_FILE.exists():
    print("=" * 64)
    print("WARNING: models/model.pkl not found.")
    print("The web app will start, but predictions will fail until you run:")
    print("    python src/train_models.py")
    print("(and first: python data/generate_synthetic_data.py)")
    print("=" * 64)

from web.app import app  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nOptiCrop starting at http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
