from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

from .training import MemoryBoundedTrainer, TrainingBatchStats


@dataclass(frozen=True)
class TrainingRunResult:
    status: str
    batches: int
    rows: int
    peak_rows_per_batch: int


class TrainingOrchestrator:
    """
    Coordinates memory-bounded batch training.

    This layer owns orchestration only. It does not select models,
    alter predictor logic, retrain existing production models, or
    accumulate the complete dataset.
    """

    def __init__(
        self,
        fit_batch: Callable[[pd.DataFrame], object],
    ) -> None:
        self._trainer = MemoryBoundedTrainer(fit_batch)

    def run(
        self,
        batches: Iterable[pd.DataFrame],
    ) -> TrainingRunResult:
        stats: TrainingBatchStats = self._trainer.fit_batches(batches)

        return TrainingRunResult(
            status="completed",
            batches=stats.batches,
            rows=stats.rows,
            peak_rows_per_batch=stats.peak_rows_per_batch,
        )
