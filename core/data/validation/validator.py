from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.data.dataset import UniversalDataset


@dataclass(frozen=True)
class DataValidationResult:
    valid: bool
    rows: int
    columns: int
    missing_cells: int
    duplicate_rows: int
    issues: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_dataset(
    dataset: UniversalDataset | pd.DataFrame,
    *,
    allow_empty: bool = False,
) -> DataValidationResult:
    """
    Validate the canonical UniversalDataset without modifying it.

    This is intentionally source-agnostic:
    CSV, Excel, SQLite, SQL, Parquet, etc. all arrive here
    through the same UniversalDataset contract.
    """
    # Backward-compatible DataFrame entry point.
    # The canonical contract remains UniversalDataset internally.
    if isinstance(dataset, pd.DataFrame):
        dataset = UniversalDataset(
            data=dataset,
            source_type="dataframe",
        )

    if not isinstance(dataset, UniversalDataset):
        raise TypeError(
            "dataset must be a UniversalDataset or pandas DataFrame"
        )

    frame = dataset.data

    issues: list[str] = []

    rows = int(len(frame))
    columns = int(len(frame.columns))
    missing_cells = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())

    if not allow_empty and rows == 0:
        issues.append("dataset contains no rows")

    if columns == 0:
        issues.append("dataset contains no columns")

    if frame.columns.duplicated().any():
        issues.append("dataset contains duplicate column names")

    empty_names = [
        str(name)
        for name in frame.columns
        if str(name).strip() == ""
    ]

    if empty_names:
        issues.append("dataset contains empty column names")

    return DataValidationResult(
        valid=len(issues) == 0,
        rows=rows,
        columns=columns,
        missing_cells=missing_cells,
        duplicate_rows=duplicate_rows,
        issues=tuple(issues),
        metadata={
            "source_type": dataset.source_type,
            "source": dataset.source,
        },
    )

def duplicate_rows(
    dataset: UniversalDataset | pd.DataFrame,
) -> int:
    """
    Return duplicate row count.

    Supports the canonical UniversalDataset contract and the
    historical pandas.DataFrame calling convention.
    """
    if isinstance(dataset, pd.DataFrame):
        return int(dataset.duplicated().sum())

    if not isinstance(dataset, UniversalDataset):
        raise TypeError(
            "dataset must be a UniversalDataset or pandas DataFrame"
        )

    return int(dataset.data.duplicated().sum())


def missingness(
    dataset: UniversalDataset | pd.DataFrame,
):
    """
    Return missingness information.

    UniversalDataset:
        Total missing cell count, preserving the Phase 5 canonical API.

    pandas.DataFrame:
        Per-column missing proportion, preserving the historical API.
    """
    if isinstance(dataset, pd.DataFrame):
        return dataset.isna().mean()

    if not isinstance(dataset, UniversalDataset):
        raise TypeError(
            "dataset must be a UniversalDataset or pandas DataFrame"
        )

    return int(dataset.data.isna().sum().sum())
