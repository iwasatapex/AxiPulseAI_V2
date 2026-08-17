from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd


@dataclass(frozen=True)
class StreamStats:
    chunks: int
    rows: int
    columns: int
    peak_rows_per_chunk: int


def process_chunks(
    chunks: Iterable[pd.DataFrame],
    processor: Callable[[pd.DataFrame], object],
) -> StreamStats:
    """
    Process chunks one at a time.

    The processor receives only the current chunk. No concatenation
    or full-dataset accumulation occurs here.
    """
    if not callable(processor):
        raise TypeError("processor must be callable")

    chunk_count = 0
    row_count = 0
    column_count = 0
    peak_rows = 0

    for frame in chunks:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("each chunk must be a pandas DataFrame")

        processor(frame)

        chunk_count += 1
        row_count += int(len(frame))
        column_count = max(column_count, int(len(frame.columns)))
        peak_rows = max(peak_rows, int(len(frame)))

    return StreamStats(
        chunks=chunk_count,
        rows=row_count,
        columns=column_count,
        peak_rows_per_chunk=peak_rows,
    )
