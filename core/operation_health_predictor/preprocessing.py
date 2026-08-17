"""
Dataset loading and feature-statistics computation.

Moved verbatim out of OperationalHealthPredictor in operation_health_predictor.py
(Phase 2, Step 2 — no logic changed).
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .constants import ISSUE_PREFIX, REQUIRED_COLUMNS
from .utils import logger


class DataLoadingMixin:
    # ---------------------------------------------------------------
    # Data Loading
    # ---------------------------------------------------------------
    def load_data(self, filepath: Union[str, Path]):
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        loaders = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
        }
        if path.suffix.lower() not in loaders:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        df = loaders[path.suffix.lower()](path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

        if "date" not in df.columns:
            for c in df.columns:
                if c.lower() in ("date", "day", "datetime"):
                    df.rename(columns={c: "date"}, inplace=True)
                    break
            else:
                raise ValueError("Dataset must contain a 'date' column.")

        raw_dates = df["date"].copy()

        # format="mixed" makes pandas infer the date format per-row instead
        # of locking onto the format of the first value and applying it to
        # the whole column (which silently NaTs rows in a different format,
        # e.g. some rows with a time component and some without).
        try:
            parsed = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
        except TypeError:
            # pandas < 2.0 doesn't support format="mixed"; fall back to
            # per-element parsing.
            parsed = raw_dates.apply(lambda v: pd.to_datetime(v, errors="coerce"))

        # Fallback: retry only the still-unparsed values with dayfirst=True,
        # in case the file uses DD/MM/YYYY instead of MM/DD/YYYY.
        still_bad = parsed.isna() & raw_dates.notna()
        if still_bad.any():
            try:
                retry = pd.to_datetime(raw_dates[still_bad], errors="coerce", dayfirst=True, format="mixed")
            except TypeError:
                retry = raw_dates[still_bad].apply(lambda v: pd.to_datetime(v, errors="coerce", dayfirst=True))
            parsed.loc[still_bad] = retry

        df["date"] = parsed

        if df["date"].isna().any():
            bad_mask = df["date"].isna()
            bad_rows = [
                f"row {i} -> {raw_dates.loc[i]!r}"
                for i in df.index[bad_mask][:20]
            ]
            n_bad = int(bad_mask.sum())
            more = f" (+{n_bad - len(bad_rows)} more)" if n_bad > len(bad_rows) else ""
            raise ValueError(
                f"{n_bad} date value(s) could not be parsed (check format). "
                f"Offending values: {'; '.join(bad_rows)}{more}"
            )

        # Repeated dates are independent simulation samples. Never aggregate
        # them by date; doing so destroys trajectory identity and leaks across
        # samples. Temporal alignment is handled by common.temporal_dataset.

        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        numeric_cols = [
            "target_quality", "actual_quality",
            "target_competency", "actual_competency",
            "target_attendance", "actual_attendance",
            "target_release_rate", "actual_release_rate",
            "target_transfer_rate", "actual_transfer_rate",
            "total_calls_received", "operational_intelligence_factor", "operational_health",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        issue_cols = [c for c in df.columns if c.startswith(ISSUE_PREFIX)]
        for col in issue_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        if issue_cols:
            total = df[issue_cols].sum(axis=1)
            if not np.all(np.isclose(total, 100, atol=1.0)):
                logger.warning("Issue type percentages do not sum to 100; they will be treated as counts.")

        self._issue_cols = issue_cols
        return df, issue_cols

    def _compute_feature_stats(self, X: pd.DataFrame):
        stats = {}
        for col in X.columns:
            vals = X[col].dropna()
            if len(vals) > 10:
                stats[f"{col}_median"] = np.median(vals)
                stats[f"{col}_q1"] = np.percentile(vals, 25)
                stats[f"{col}_q3"] = np.percentile(vals, 75)
                stats[f"{col}_std"] = vals.std()
            else:
                stats[f"{col}_median"] = vals.median()
                stats[f"{col}_q1"] = vals.min()
                stats[f"{col}_q3"] = vals.max()
                stats[f"{col}_std"] = vals.std() if len(vals) > 1 else 1.0
        return stats


def load_data(predictor, filepath):
    """Compatibility API for DataLoadingMixin.load_data()."""
    if predictor is None:
        raise TypeError(
            "load_data() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return DataLoadingMixin.load_data(
        predictor,
        filepath,
    )
