"""
Module: shap_explainer.py

Purpose:
Generate SHAP explanations for the trained Random Forest model.

Author:
Swarup Pingle

Project:
CardioSense AI
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "processed"
FIGURE_PATH = BASE_DIR / "reports" / "figures"

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load Model and Data
# ==========================================================

model = joblib.load(MODEL_PATH / "random_forest.pkl")

X_train = pd.read_csv(DATA_PATH / "X_train.csv")

# ==========================================================
# SHAP Explainer
# ==========================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_train)

# ==========================================================
# Summary Plot
# ==========================================================

plt.figure(figsize=(10, 6))

# For binary classification Random Forest
shap.summary_plot(
    shap_values[:, :, 1],
    X_train,
    show=False
)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("CardioSense AI - SHAP Analysis")
print("=" * 60)
print("✓ SHAP Summary Plot Generated")
print(f"Saved to: {FIGURE_PATH}")

# ==========================================================
# Explain One Patient
# ==========================================================

patient_index = 0

print("\nGenerating Local Explanation...")

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[patient_index, :, 1],
        base_values=explainer.expected_value[1],
        data=X_train.iloc[patient_index],
        feature_names=X_train.columns,
    ),
    show=False,
)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "patient_explanation.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("✓ Patient Explanation Saved")