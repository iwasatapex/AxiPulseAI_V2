from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class UniversalDataset:
    """
    Canonical dataset representation for AxiPulseAI.

    Source-specific ingestion belongs outside this object.
    Predictors consume the canonical dataframe without needing to know
    whether the source was CSV, Excel, SQLite, SQL, or another adapter.
    """

    data: pd.DataFrame
    source_type: str
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if not self.source_type:
            raise ValueError("source_type must not be empty")

    @property
    def rows(self) -> int:
        return len(self.data)

    @property
    def columns(self) -> list[str]:
        return list(self.data.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def copy(self) -> "UniversalDataset":
        return UniversalDataset(
            data=self.data.copy(),
            source_type=self.source_type,
            source=self.source,
            metadata=dict(self.metadata),
        )
