from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import pandas as pd


@dataclass(frozen=True)
class TrainingBatchStats:
    batches: int
    rows: int
    peak_rows_per_batch: int


class MemoryBoundedTrainer:
    """
    Generic batch-oriented training interface.

    The trainer delegates model-specific learning to the supplied
    callback and never accumulates the complete dataset in memory.
    """

    def __init__(
        self,
        fit_batch: Callable[[pd.DataFrame], Any],
    ) -> None:
        if not callable(fit_batch):
            raise TypeError("fit_batch must be callable")

        self._fit_batch = fit_batch

    def fit_batches(
        self,
        batches: Iterable[pd.DataFrame],
    ) -> TrainingBatchStats:
        batch_count = 0
        row_count = 0
        peak_rows = 0

        for frame in batches:
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    "each training batch must be a pandas DataFrame"
                )

            self._fit_batch(frame.copy())

            batch_count += 1
            row_count += len(frame)
            peak_rows = max(peak_rows, len(frame))

        return TrainingBatchStats(
            batches=batch_count,
            rows=row_count,
            peak_rows_per_batch=peak_rows,
        )
