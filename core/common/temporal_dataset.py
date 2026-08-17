"""Temporal dataset alignment helpers for the forecasting contract.

Forecasting contract:

    feature_time[T] < target_time[T+1]

Features observed at time T predict the target realized at T+1. Trainers
therefore must align each feature row with the NEXT row's target. The final
historical row has no T+1 target and must be excluded from training.

This module is the single shared implementation for that forward-shift; both
the OH and NPS trainers call :func:`shift_target_next_day` so the temporal
alignment stays consistent across engines.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import pandas as pd

from .temporal_contract import assert_forecast_boundary


def _coerce_series(values, name):
    if values is None:
        return None
    return pd.Series(values).reset_index(drop=True).rename(name)


def _validate_monotonic_dates(times: pd.Series) -> None:
    if not times.is_monotonic_increasing:
        raise ValueError(
            "Temporal dataset must be sorted by date before target alignment; "
            "non-monotonic dates would make T+1 ambiguous."
        )


def shift_target_next_day(
    target: Union[pd.Series, pd.DataFrame],
    times: Sequence,
    *,
    trajectory_ids: Sequence | None = None,
    field_name: str = "target",
) -> tuple:
    """Align each feature row with the target on the next calendar date.

    For repeated-date simulation data, ``trajectory_ids`` is preferred and
    pairs ``(trajectory_id, T)`` with ``(trajectory_id, T+1)``.  When no
    trajectory id is supplied, the function uses the stable occurrence index
    within each date as the simulation stream key.  This fallback is retained
    for legacy date-major simulation files, but explicit trajectory IDs are
    recommended whenever available.
    """
    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    target = target.reset_index(drop=True)
    n = len(times)
    if len(target) != n:
        raise ValueError(f"{field_name}: target and times must have equal length")
    _validate_monotonic_dates(times)

    ids = _coerce_series(trajectory_ids, "trajectory_id")
    if ids is not None and len(ids) != n:
        raise ValueError(f"{field_name}: trajectory_ids and times must have equal length")

    repeated = times.duplicated().any()
    if not repeated:
        shifted = target.shift(-1)
        target_times = times.shift(-1)
    else:
        if ids is None:
            # Legacy-safe fallback: occurrence k on date T is paired with
            # occurrence k on T+1. This is deterministic and never crosses a
            # date boundary incorrectly, but cannot prove simulation identity.
            stream = times.groupby(times, sort=False).cumcount()
        else:
            # Explicit trajectory IDs are the authoritative identity.
            if ids.isna().any():
                raise ValueError(f"{field_name}: trajectory_ids contains missing values")
            dup = pd.DataFrame({"trajectory_id": ids, "date": times}).duplicated()
            if dup.any():
                raise ValueError(
                    f"{field_name}: duplicate rows for the same trajectory_id/date; "
                    "temporal alignment is ambiguous"
                )
            stream = ids.astype(str)

        work = pd.DataFrame({"orig_idx": range(n), "date": times, "stream": stream})
        unique_dates = pd.Index(times.drop_duplicates())
        date_rank = {d: i for i, d in enumerate(unique_dates)}
        work["rank"] = work["date"].map(date_rank).astype(int)

        work = work.sort_values(["stream", "rank", "orig_idx"], kind="stable")
        work["next_orig_idx"] = work.groupby("stream", sort=False)["orig_idx"].shift(-1)
        work["next_rank"] = work.groupby("stream", sort=False)["rank"].shift(-1)
        work["target_orig_idx"] = work["next_orig_idx"].where(
            work["next_rank"].eq(work["rank"] + 1)
        )
        work = work.sort_values("orig_idx", kind="stable")

        target_times = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
        shifted = target.copy(deep=True)
        if isinstance(target, pd.DataFrame):
            shifted = pd.DataFrame(index=range(n), columns=target.columns, dtype="float64")
        else:
            shifted = pd.Series(float("nan"), index=range(n), dtype="float64")

        valid = work["target_orig_idx"].notna()
        src = work.loc[valid, "target_orig_idx"].astype(int).to_numpy()
        dst = work.index[valid].to_numpy()
        target_times.iloc[dst] = times.iloc[src].to_numpy()
        if isinstance(target, pd.DataFrame):
            shifted.iloc[dst, :] = target.iloc[src].to_numpy()
        else:
            shifted.iloc[dst] = target.iloc[src].to_numpy()

    if isinstance(shifted, pd.DataFrame):
        has_target = shifted.notna().all(axis=1)
    else:
        has_target = shifted.notna()
    valid = target_times.notna() & has_target
    for i in times.index[valid]:
        assert_forecast_boundary(times[i], target_times[i])
    return shifted, target_times


def date_aware_splits(times: Sequence, n_splits: int = 5):
    """Yield train/validation row indices with whole calendar dates held out."""
    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    unique = pd.Index(times.drop_duplicates())
    if len(unique) < 2:
        raise ValueError("Need at least two distinct dates for temporal validation")
    n_splits = min(int(n_splits), len(unique) - 1)
    if n_splits < 2:
        # One deterministic holdout when the history is very short.
        cut = max(1, len(unique) - 1)
        train_dates = set(unique[:cut])
        val_dates = set(unique[cut:])
        train_idx = times.index[times.isin(train_dates)].to_numpy()
        val_idx = times.index[times.isin(val_dates)].to_numpy()
        yield train_idx, val_idx
        return
    # Expanding-window splits using contiguous date blocks.
    fold_size = max(1, len(unique) // (n_splits + 1))
    for fold in range(1, n_splits + 1):
        val_start = fold * fold_size
        val_end = len(unique) if fold == n_splits else min(len(unique), val_start + fold_size)
        if val_start >= val_end:
            continue
        train_dates = set(unique[:val_start])
        val_dates = set(unique[val_start:val_end])
        train_idx = times.index[times.isin(train_dates)].to_numpy()
        val_idx = times.index[times.isin(val_dates)].to_numpy()
        if len(train_idx) and len(val_idx):
            yield train_idx, val_idx


def tail_by_distinct_dates(df: pd.DataFrame, date_col: str, days: int) -> pd.DataFrame:
    """Return the last N distinct calendar dates, preserving all rows."""
    if df.empty or days <= 0:
        return df.iloc[0:0].copy()
    dates = pd.to_datetime(df[date_col])
    keep = pd.Index(dates.drop_duplicates().sort_values())[-int(days):]
    return df.loc[dates.isin(keep)].copy()
