"""
Module: loader.py

Purpose:
Load the UCI Heart Disease dataset.

Author:
Swarup Pingle

Project:
CardioSense AI
"""

from pathlib import Path
import pandas as pd

# Official UCI Cleveland dataset column names
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

DATASET_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "raw"
    / "uci-heart-disease"
    / "processed.cleveland.data"
)


def load_dataset(file_path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load the heart disease dataset.

    Args:
        file_path: Path to the dataset file.

    Returns:
        Loaded pandas DataFrame.
    """

    try:
        dataset = pd.read_csv(
            file_path,
            header=None,
            names=COLUMN_NAMES,
            na_values="?"
        )

        return dataset

    except FileNotFoundError:
        print(f"Dataset not found: {file_path}")
        raise

    except Exception as error:
        print(f"Unexpected error: {error}")
        raise