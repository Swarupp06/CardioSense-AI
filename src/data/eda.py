"""
Module: eda.py

Purpose:
Perform Exploratory Data Analysis (EDA) on the UCI Heart Disease dataset.

Author:
Swarup Pingle

Project:
CardioSense AI
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.loader import load_dataset

# ==========================================================
# Constants
# ==========================================================

FIGURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "figures"
)

FIGURE_PATH.mkdir(parents=True, exist_ok=True)


def basic_information(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    """

    print("\nDataset Shape:", df.shape)
    df.info()


def missing_value_report(df: pd.DataFrame) -> None:
    """
    Display missing values for each column.
    """

    print("\nMissing Values")
    print(df.isnull().sum())


def duplicate_report(df: pd.DataFrame) -> None:
    """
    Display duplicate row count.
    """

    print("\nDuplicate Rows:", df.duplicated().sum())


def target_distribution(df: pd.DataFrame) -> None:
    """
    Plot target distribution.
    """

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")

    plt.title("Target Distribution")

    plt.savefig(FIGURE_PATH / "target_distribution.png")

    plt.close()


def histograms(df: pd.DataFrame) -> None:
    """
    Plot histograms for numerical features.
    """

    df.hist(figsize=(15, 12))

    plt.tight_layout()

    plt.savefig(FIGURE_PATH / "histograms.png")

    plt.close()


def correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot correlation heatmap.
    """

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(FIGURE_PATH / "correlation_heatmap.png")

    plt.close()


def boxplots(df: pd.DataFrame) -> None:
    """
    Plot boxplots for all numerical features.
    """

    df.plot(
        kind="box",
        subplots=True,
        layout=(4, 4),
        figsize=(15, 12),
    )

    plt.tight_layout()

    plt.savefig(FIGURE_PATH / "boxplots.png")

    plt.close()


def main() -> None:
    """
    Main function.
    """

    df = load_dataset()

    basic_information(df)

    missing_value_report(df)

    duplicate_report(df)

    target_distribution(df)

    histograms(df)

    correlation_heatmap(df)

    boxplots(df)


if __name__ == "__main__":
    main()