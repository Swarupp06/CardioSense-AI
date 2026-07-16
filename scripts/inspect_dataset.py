"""
Module: inspect_dataset.py

Purpose:
Inspect the UCI Heart Disease dataset before preprocessing.

Author:
Swarup Pingle

Project:
CardioSense AI
"""

import pandas as pd
from pathlib import Path

# ==========================================================
# Constants
# ==========================================================

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
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "uci-heart-disease"
    / "processed.cleveland.data"
)

def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load the heart disease dataset.

    Args:
        file_path: Path to the dataset file.

    Returns:
        pandas.DataFrame: Loaded dataset.
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

def preview_dataset(dataset: pd.DataFrame, rows: int = 5) -> None:
    """
    Display the first few rows of the dataset.

    Args:
        dataset: Loaded dataset.
        rows: Number of rows to display.
    """

    print("\n" + "=" * 60)
    print("DATASET PREVIEW")
    print("=" * 60)

    print(dataset.head(rows))

def display_dataset_info(dataset: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.

    Args:
        dataset: Loaded dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"\nNumber of Rows    : {dataset.shape[0]}")
    print(f"Number of Columns : {dataset.shape[1]}")

    print("\nData Types")
    print("-" * 60)
    print(dataset.dtypes)

    print("\nMissing Values")
    print("-" * 60)
    print(dataset.isnull().sum())

    print("\nMemory Usage")
    print("-" * 60)
    dataset.info()   

def display_summary_statistics(dataset: pd.DataFrame) -> None:
    """
    Display summary statistics for numerical features.

    Args:
        dataset: Loaded dataset.
    """

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(dataset.describe())

def main() -> None:
    """
    Main function for dataset inspection.
    """

    dataset = load_dataset(DATASET_PATH)

    preview_dataset(dataset)

    display_dataset_info(dataset)

    display_summary_statistics(dataset)  

if __name__ == "__main__":
    main()          
