from __future__ import annotations

import pandas as pd


def validate_dataset(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame")

    if data.empty:
        raise ValueError("dataset must not be empty")

    if len(set(data.columns)) != len(data.columns):
        raise ValueError("dataset contains duplicate column names")


def missingness(data: pd.DataFrame) -> dict[str, float]:
    validate_dataset(data)

    return {
        str(column): float(data[column].isna().mean())
        for column in data.columns
    }


def duplicate_rows(data: pd.DataFrame) -> int:
    validate_dataset(data)
    return int(data.duplicated().sum())
