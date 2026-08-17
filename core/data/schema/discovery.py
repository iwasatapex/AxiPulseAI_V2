from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


ColumnKind = Literal[
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "text",
    "identifier",
]


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    kind: ColumnKind
    dtype: str
    nullable: bool


def infer_column_kind(
    series: pd.Series,
) -> ColumnKind:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        name = str(series.name).strip().lower()

        identifier_names = {
            "id",
            "identifier",
            "uuid",
            "guid",
        }

        if (
            name in identifier_names
            or name.endswith("_id")
            or name.endswith("-id")
            or name.endswith(" id")
        ):
            return "identifier"

        return "numeric"

    unique = series.nunique(dropna=True)
    total = len(series.dropna())

    if total > 0 and unique / total > 0.95:
        return "identifier"

    if pd.api.types.is_object_dtype(series):
        average_length = (
            series.dropna()
            .astype(str)
            .str.len()
            .mean()
            if total
            else 0
        )

        if average_length > 40:
            return "text"

    return "categorical"


def discover_schema(
    data: pd.DataFrame,
) -> list[ColumnSchema]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    return [
        ColumnSchema(
            name=str(name),
            kind=infer_column_kind(data[name]),
            dtype=str(data[name].dtype),
            nullable=bool(data[name].isna().any()),
        )
        for name in data.columns
    ]
