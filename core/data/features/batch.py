from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd


@dataclass(frozen=True)
class BatchFeatureStats:
    batches: int
    rows: int
    columns: int
    peak_rows_per_batch: int


def process_feature_batches(
    batches: Iterable[pd.DataFrame],
    transformer: Callable[[pd.DataFrame], pd.DataFrame],
) -> BatchFeatureStats:
    """
    Apply feature transformation to one dataframe batch at a time.

    The transformed batch is intentionally not accumulated here.
    This keeps memory usage proportional to the active batch rather
    than the complete dataset.
    """
    if not callable(transformer):
        raise TypeError("transformer must be callable")

    batch_count = 0
    row_count = 0
    column_count = 0
    peak_rows = 0

    for frame in batches:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "each feature batch must be a pandas DataFrame"
            )

        transformed = transformer(frame)

        if not isinstance(transformed, pd.DataFrame):
            raise TypeError(
                "feature transformer must return a pandas DataFrame"
            )

        if len(transformed) != len(frame):
            raise ValueError(
                "feature transformer must preserve row count"
            )

        batch_count += 1
        row_count += len(transformed)
        column_count = max(column_count, len(transformed.columns))
        peak_rows = max(peak_rows, len(transformed))

    return BatchFeatureStats(
        batches=batch_count,
        rows=row_count,
        columns=column_count,
        peak_rows_per_batch=peak_rows,
    )
