"""Model building & evaluation for OptiCrop (Epic 4).

Pipeline
--------
1. Load + preprocess data (reuses `src/preprocess.py`)
2. K-Means clustering with the Elbow Method (unsupervised pattern discovery)
3. Train supervised classifiers:
       Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest
4. Evaluate each with Accuracy / Precision / Recall / F1 (macro) + confusion matrix
5. Pick the best model by macro-F1, bundle it (scaler + model + label encoder)
   and pickle it to ``models/model.pkl`` for the web app.

Run:
    python src/train_models.py
"""
import pickle
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.preprocess import build_pipeline  # noqa: E402

warnings.filterwarnings("ignore", message="Unknown solver options: iprint")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# K-Means + Elbow Method
# ---------------------------------------------------------------------------
def run_kmeans_elbow(X: np.ndarray) -> None:
    """Plot the elbow curve to choose k, then fit final K-Means."""
    wcss = []
    K_RANGE = range(2, 13)
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE)
        km.fit(X)
        wcss.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(K_RANGE), wcss, marker="o", color="#2a9d8f")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.title("K-Means Elbow Method")
    plt.xticks(list(K_RANGE))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURE_DIR / "kmeans_elbow.png", dpi=120)
    plt.close()
    print(f"[KMeans] Elbow plot saved to {config.FIGURE_DIR / 'kmeans_elbow.png'}")

    # Fit final K-Means at k=8 (a reasonable elbow for ~22 crops)
    final_k = 8
    km = KMeans(n_clusters=final_k, n_init=10, random_state=config.RANDOM_STATE)
    labels = km.fit_predict(X)
    print(f"[KMeans] Final clustering with k={final_k}: "
          f"{np.unique(labels, return_counts=True)}")


# ---------------------------------------------------------------------------
# Supervised models
# ---------------------------------------------------------------------------
def build_classifiers() -> dict:
    """Instantiate all classifiers used in the comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=config.RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(
            random_state=config.RANDOM_STATE, max_depth=12),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=1),
    }


def evaluate(model, X_te, y_te) -> dict:
    pred = model.predict(X_te)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, average="macro", zero_division=0),
        "recall": recall_score(y_te, pred, average="macro", zero_division=0),
        "f1": f1_score(y_te, pred, average="macro", zero_division=0),
        "confusion": confusion_matrix(y_te, pred),
        "report": classification_report(y_te, pred, zero_division=0),
    }


def save_confusion(cm: np.ndarray, classes: np.ndarray, name: str) -> None:
    import seaborn as sns
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    safe = name.lower().replace(" ", "_")
    plt.savefig(config.FIGURE_DIR / f"confusion_{safe}.png", dpi=110)
    plt.close()


def main() -> None:
    print("=" * 70)
    print("OptiCrop - Model Training & Evaluation")
    print("=" * 70)

    X_train, X_test, y_train, y_test = build_pipeline()

    # Scale features — required for LogReg/KNN and improves the others
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 1) Unsupervised: K-Means + elbow
    print("\n--- K-Means clustering ---")
    run_kmeans_elbow(X_train_s)

    # 2) Supervised classifiers
    print("\n--- Supervised model comparison ---")
    results = {}
    for name, model in build_classifiers().items():
        model.fit(X_train_s, y_train)
        m = evaluate(model, X_test_s, y_test)
        results[name] = (model, m)
        print(f"\n{name}: accuracy={m['accuracy']:.4f}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"f1={m['f1']:.4f}")
        save_confusion(m["confusion"], np.unique(y_train), name)

    # 3) Pick the best model by macro-F1
    best_name = max(results, key=lambda n: results[n][1]["f1"])
    best_model, best_metrics = results[best_name]
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name}  (macro-F1 = {best_metrics['f1']:.4f})")
    print("=" * 70)
    print(best_metrics["report"])

    # 4) Save bundle: scaler + model + metadata
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "feature_columns": config.FEATURE_COLUMNS,
        "target_column": config.TARGET_COLUMN,
        "model_name": best_name,
        "metrics": {
            "accuracy": float(best_metrics["accuracy"]),
            "precision": float(best_metrics["precision"]),
            "recall": float(best_metrics["recall"]),
            "f1": float(best_metrics["f1"]),
        },
        "all_results": {n: {k: float(v) for k, v in r[1].items()
                            if isinstance(v, float) or k in ("accuracy",)}
                        for n, r in results.items()},
    }
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[OK] Saved best model bundle to {config.MODEL_FILE}")

    # Save a small metrics table for the README
    metrics_df = pd.DataFrame([
        {"model": n, **{k: v for k, v in r[1].items()
                        if k in ("accuracy", "precision", "recall", "f1")}}
        for n, r in results.items()
    ]).round(4)
    metrics_df.to_csv(config.OUTPUT_DIR / "model_metrics.csv", index=False)
    print(f"[OK] Metrics table saved to {config.OUTPUT_DIR / 'model_metrics.csv'}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
