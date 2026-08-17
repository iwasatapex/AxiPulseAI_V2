from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.data import UniversalDataset


def _dataset_from_frame(
    frame: pd.DataFrame,
    source_type: str,
) -> UniversalDataset:
    """
    Convert a loaded pandas DataFrame into the existing
    UniversalDataset contract without changing the dataset model.
    """
    return UniversalDataset(
        data=frame,
        source_type=source_type,
    )


def load_excel(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    **kwargs: Any,
) -> UniversalDataset:
    """Load an Excel worksheet into UniversalDataset."""
    frame = pd.read_excel(
        Path(path),
        sheet_name=sheet_name,
        **kwargs,
    )
    return _dataset_from_frame(frame, "excel")


def load_sqlite(
    path: str | Path,
    query: str,
    *,
    params: Any = None,
) -> UniversalDataset:
    """Load a SQLite query result into UniversalDataset."""
    import sqlite3

    database = Path(path)

    with sqlite3.connect(database) as connection:
        frame = pd.read_sql_query(
            query,
            connection,
            params=params,
        )

    return _dataset_from_frame(frame, "sqlite")
