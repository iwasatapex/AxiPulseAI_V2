from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..dataset import UniversalDataset


def load_csv(
    path: str | Path,
    **read_csv_kwargs,
) -> UniversalDataset:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    data = pd.read_csv(path, **read_csv_kwargs)

    return UniversalDataset(
        data=data,
        source_type="csv",
        source=str(path),
        metadata={
            "reader": "pandas.read_csv",
        },
    )
