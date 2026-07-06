"""OptiCrop Flask web application (Epic 5).

Serves three pages — Home, About, FindYourCrop — and an endpoint that runs the
trained model on user-supplied soil & climate parameters.

Run from the project root:
    python run.py
or directly:
    python web/app.py
"""
import os
import pickle
import sys
from pathlib import Path

from flask import Flask, render_template, request

# Make project root importable so `import config` works regardless of CWD
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from src.preprocess import transform_prediction_features  # noqa: E402

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.config["SECRET_KEY"] = os.environ.get("OPTICROP_SECRET", "opticrop-dev-key")

# ---------------------------------------------------------------------------
# Load the trained model bundle once at startup
# ---------------------------------------------------------------------------
def load_bundle():
    if not config.MODEL_FILE.exists():
        return None
    with open(config.MODEL_FILE, "rb") as f:
        return pickle.load(f)


BUNDLE = load_bundle()

# Field definitions for the prediction form (label, key, unit, min, max, step, default)
FORM_FIELDS = [
    {"key": "N",           "label": "Nitrogen (N)",       "unit": "kg/ha",  "min": 0,   "max": 150, "step": 1,   "value": 80},
    {"key": "P",           "label": "Phosphorous (P)",    "unit": "kg/ha",  "min": 0,   "max": 150, "step": 1,   "value": 48},
    {"key": "K",           "label": "Potassium (K)",      "unit": "kg/ha",  "min": 0,   "max": 210, "step": 1,   "value": 40},
    {"key": "temperature", "label": "Temperature",        "unit": "°C",     "min": 5,   "max": 50,  "step": 0.1, "value": 23.0},
    {"key": "humidity",    "label": "Humidity",           "unit": "%",      "min": 10,  "max": 100, "step": 0.1, "value": 82.0},
    {"key": "ph",          "label": "Soil pH",            "unit": "",       "min": 3.5, "max": 10,  "step": 0.1, "value": 6.4},
    {"key": "rainfall",    "label": "Rainfall",           "unit": "mm",     "min": 10,  "max": 300, "step": 1,   "value": 225},
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html", model_ready=BUNDLE is not None)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/find-your-crop", methods=["GET", "POST"])
def find_your_crop():
    prediction = None
    error = None
    inputs = {f["key"]: f["value"] for f in FORM_FIELDS}

    if request.method == "POST":
        try:
            if BUNDLE is None:
                raise RuntimeError("Model not trained. Run `python src/train_models.py`.")
            for field in FORM_FIELDS:
                inputs[field["key"]] = float(request.form.get(field["key"], field["value"]))
            row = [transform_prediction_features(inputs, BUNDLE["feature_columns"])]
            scaled = BUNDLE["scaler"].transform(row)
            prediction = str(BUNDLE["model"].predict(scaled)[0]).capitalize()
        except (ValueError, TypeError):
            error = "Please enter valid numeric values for all fields."
        except RuntimeError as e:
            error = str(e)

    return render_template("find_your_crop.html",
                           fields=FORM_FIELDS, inputs=inputs,
                           prediction=prediction, error=error)


@app.errorhandler(404)
def not_found(_):
    return render_template("home.html", model_ready=BUNDLE is not None), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
