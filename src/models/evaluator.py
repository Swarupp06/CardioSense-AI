"""
Module: evaluator.py

Purpose:
Evaluate all trained machine learning models and compare their performance.

Author:
Swarup Pingle

Project:
CardioSense AI
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"
REPORT_PATH = BASE_DIR / "reports" / "metrics"

REPORT_PATH.mkdir(parents=True, exist_ok=True)


def load_test_data():
    """
    Load processed testing dataset.
    """

    X_test = pd.read_csv(DATA_PATH / "X_test.csv")
    y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

    return X_test, y_test


def load_models():
    """
    Load every trained model.
    """

    models = {}

    for model_file in MODEL_PATH.glob("*.pkl"):
        models[model_file.stem] = joblib.load(model_file)

    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate every model.
    """

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = None

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc,
            }
        )

    return pd.DataFrame(results)


def main():

    print("=" * 60)
    print("CardioSense AI - Model Evaluation")
    print("=" * 60)

    X_test, y_test = load_test_data()

    models = load_models()

    results = evaluate_models(models, X_test, y_test)

    results = results.sort_values(
        by="Accuracy",
        ascending=False,
    )

    print("\nModel Comparison\n")
    print(results)

    results.to_csv(
        REPORT_PATH / "model_comparison.csv",
        index=False,
    )

    print("\n✓ Metrics saved successfully.")


if __name__ == "__main__":
    main()