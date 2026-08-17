from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd

from .training_orchestration import TrainingOrchestrator, TrainingRunResult


@dataclass(frozen=True)
class LargeDataTrainingConfig:
    """
    Configuration for memory-bounded large-data training.

    The adapter controls data access and batching only.
    It does not retrain or replace production models by itself.
    """

    chunksize: int = 10000

    def __post_init__(self) -> None:
        if self.chunksize <= 0:
            raise ValueError("chunksize must be positive")


class LargeDataTrainingAdapter:
    """
    Universal adapter for large tabular datasets.

    Supported source:
      - CSV

    The interface is intentionally extensible for future:
      - SQLite
      - SQL
      - Parquet
      - other streaming-capable sources
    """

    def __init__(
        self,
        fit_batch,
        *,
        config: LargeDataTrainingConfig | None = None,
    ) -> None:
        if not callable(fit_batch):
            raise TypeError("fit_batch must be callable")

        self.config = config or LargeDataTrainingConfig()
        self._orchestrator = TrainingOrchestrator(fit_batch)

    def csv_batches(
        self,
        path: str | Path,
        **read_csv_kwargs: Any,
    ) -> Iterator[pd.DataFrame]:
        """
        Stream CSV rows as bounded pandas DataFrames.
        """
        csv_path = Path(path)

        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        yield from pd.read_csv(
            csv_path,
            chunksize=self.config.chunksize,
            **read_csv_kwargs,
        )

    def parquet_batches(
        self,
        path: str | Path,
        **read_parquet_kwargs: Any,
    ) -> Iterator[pd.DataFrame]:
        """
        Read a Parquet source through pandas.

        Row-group/engine-specific streaming remains delegated to the
        selected Parquet backend. No full-file accumulation is performed
        by this adapter.
        """
        parquet_path = Path(path)

        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)

        frame = pd.read_parquet(
            parquet_path,
            **read_parquet_kwargs,
        )

        for start in range(0, len(frame), self.config.chunksize):
            yield frame.iloc[
                start:start + self.config.chunksize
            ].copy()

    def train_parquet(
        self,
        path: str | Path,
        **read_parquet_kwargs: Any,
    ) -> TrainingRunResult:
        """Run bounded batch training over a Parquet source."""
        return self._orchestrator.run(
            self.parquet_batches(
                path,
                **read_parquet_kwargs,
            )
        )

    def sql_batches(
        self,
        connection,
        query: str,
        *,
        params: Any = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Stream SQL query results using pandas chunks.

        The connection is owned by the caller.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        yield from pd.read_sql_query(
            query,
            connection,
            params=params,
            chunksize=self.config.chunksize,
        )

    def train_sql(
        self,
        connection,
        query: str,
        *,
        params: Any = None,
    ) -> TrainingRunResult:
        """Run bounded batch training over a SQL query."""
        return self._orchestrator.run(
            self.sql_batches(
                connection,
                query,
                params=params,
            )
        )

    def train_csv(
        self,
        path: str | Path,
        **read_csv_kwargs: Any,
    ) -> TrainingRunResult:
        """
        Run memory-bounded batch training over a CSV source.
        """
        return self._orchestrator.run(
            self.csv_batches(
                path,
                **read_csv_kwargs,
            )
        )
