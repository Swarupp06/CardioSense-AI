"""
Module: trainer.py

Purpose:
Train machine learning models for CardioSense AI.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"

MODEL_PATH.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    """
    Load processed train/test datasets.
    """

    X_train = pd.read_csv(DATA_PATH / "X_train.csv")
    X_test = pd.read_csv(DATA_PATH / "X_test.csv")

    y_train = pd.read_csv(DATA_PATH / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


def get_models():
    """
    Return all machine learning models.
    """

    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
        "knn": KNeighborsClassifier(),
        "svm": SVC(probability=True, random_state=42),
    }


def train_models(models, X_train, y_train):
    """
    Train every model and save it.
    """

    trained_models = {}

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

        joblib.dump(model, MODEL_PATH / f"{name}.pkl")

    return trained_models


def main():

    print("=" * 60)
    print("CardioSense AI - Model Training")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_processed_data()

    models = get_models()

    train_models(models, X_train, y_train)

    print("\nAll models trained successfully!")

    print(f"\nSaved models to:\n{MODEL_PATH}")


if __name__ == "__main__":
    main()