"""
Module: preprocessor.py

Purpose:
Preprocess the UCI Heart Disease dataset for machine learning.

Author:
Swarup Pingle

Project:
CardioSense AI
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_dataset

# ==========================================================
# Constants
# ==========================================================

PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed"
)

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
)

MODEL_PATH.mkdir(parents=True, exist_ok=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using the most frequent value.
    """

    imputer = SimpleImputer(strategy="most_frequent")

    df[["ca", "thal"]] = imputer.fit_transform(df[["ca", "thal"]])

    return df


def convert_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert target into binary classification.

    0 -> No Heart Disease
    1,2,3,4 -> Heart Disease
    """

    df["target"] = (df["target"] > 0).astype(int)

    return df


def split_dataset(df: pd.DataFrame):
    """
    Split dataset into training and testing sets.
    """

    X = df.drop(columns=["target"])
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Standardize numerical features.
    """

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
    )

    # Save fitted scaler
    joblib.dump(
        scaler,
        MODEL_PATH / "scaler.pkl"
    )

    return X_train_scaled, X_test_scaled


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Save processed datasets.
    """

    X_train.to_csv(PROCESSED_DATA_PATH / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_PATH / "X_test.csv", index=False)

    y_train.to_csv(PROCESSED_DATA_PATH / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_PATH / "y_test.csv", index=False)


def main() -> None:
    """
    Execute preprocessing pipeline.
    """

    print("=" * 60)
    print("CardioSense AI - Data Preprocessing")
    print("=" * 60)

    df = load_dataset()

    print("✓ Dataset Loaded")

    df = handle_missing_values(df)

    print("✓ Missing Values Handled")

    df = convert_target(df)

    print("✓ Target Converted")

    X_train, X_test, y_train, y_test = split_dataset(df)

    print("✓ Dataset Split")

    X_train, X_test = scale_features(
        X_train,
        X_test,
    )

    print("✓ Features Scaled")

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("✓ Processed Data Saved")

    print("\nPreprocessing Completed Successfully!")


if __name__ == "__main__":
    main()

MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
)

MODEL_PATH.mkdir(parents=True, exist_ok=True)    