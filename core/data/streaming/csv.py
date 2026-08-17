from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


@dataclass(frozen=True)
class CSVChunk:
    data: pd.DataFrame
    chunk_index: int
    source: str
    rows: int


def iter_csv_chunks(
    path: str | Path,
    *,
    chunksize: int = 100_000,
    **read_csv_kwargs: Any,
) -> Iterator[CSVChunk]:
    """
    Stream a CSV without loading the complete file into memory.

    The source file is never modified.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    reader = pd.read_csv(
        path,
        chunksize=chunksize,
        **read_csv_kwargs,
    )

    for index, frame in enumerate(reader):
        yield CSVChunk(
            data=frame,
            chunk_index=index,
            source=str(path),
            rows=int(len(frame)),
        )
