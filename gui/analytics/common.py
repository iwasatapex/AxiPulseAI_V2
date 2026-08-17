"""Shared, reusable helpers for the AxiPulseAI V2 GUI analytics layer.

This module contains only **pure, testable** analytics helpers plus thin
rendering helpers. It performs NO model / simulator mathematics — it
consumes canonical engine outputs and the canonical KPI contracts and
turns them into human-readable diagnostic structure.

Any analytical metric that cannot be derived reliably from existing engine
output is reported as unavailable rather than fabricated.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd

from gui import contracts as ct

# Marker for fields the engine does not expose.
NA = "Not available from engine output"


# ---------------------------------------------------------------------
# Safe coercion / access
# ---------------------------------------------------------------------

def fnum(value) -> Optional[float]:
    """Return a finite float or None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def disp(value: Any, nd: int = 2) -> str:
    """Format a value as a homogeneous display string.

    Returns ``"—"`` for missing/non-finite values so a dataframe column never
    mixes numeric values with strings (which Streamlit's Arrow serializer
    rejects: "object containing mixed strings/floats"). This is display-only
    and never touches model input data.
    """
    v = fnum(value)
    if v is None:
        return "—"
    return f"{v:.{int(nd)}f}"


def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts, returning ``default`` on any miss."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


# ---------------------------------------------------------------------
# KPI comparison (canonical targets, NOT simulator math)
# ---------------------------------------------------------------------

def kpi_row(label: str, key: str, value):
    """Build one KPI comparison row against the canonical target."""
    cfg = ct.KPI[key]
    target = cfg.get("target")
    lo, hi = cfg["min"], cfg["max"]
    val = fnum(value)
    met = ct.kpi_met(key, val) if target is not None else None
    gap = None
    pct = None
    if val is not None and target is not None:
        gap = target - val
        pct = (val / target * 100.0) if target else None
    return {
        "kpi": label,
        "key": key,
        "value": val,
        "unit": cfg.get("unit", ""),
        "lo": lo,
        "hi": hi,
        "target": target,
        "gap": gap,
        "pct_of_target": pct,
        "met": met,
    }


def kpi_comparison_rows(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return comparison rows for every canonical-target KPI present in state."""
    rows: List[Dict[str, Any]] = []
    for key in ct.KPI:
        if key in state and ct.KPI[key].get("target") is not None:
            rows.append(kpi_row(ct.KPI[key]["label"], key, state.get(key)))
    return rows


def day_kpi_met(day: Dict[str, Any],
                checked: Optional[List[str]] = None) -> bool:
    """A forecast day is KPI-met when >= 3 of the checked KPIs meet target.

    Default checked KPIs: quality, competency, release, transfer.  Transfer
    uses inverse semantics (lower is better).
    """
    if not isinstance(day, dict):
        return False
    checked = checked or ["quality", "competency", "release", "transfer"]
    met = [k for k in checked if ct.kpi_met(k, day.get(k)) is True]
    return len(met) >= 3


# ---------------------------------------------------------------------
# Health levels
# ---------------------------------------------------------------------

def health_level(value, good_ge: float, warn_ge: float) -> str:
    """Classify a 0-100ish score into Good / Warning / Poor / Unknown."""
    v = fnum(value)
    if v is None:
        return "Unknown"
    if v >= good_ge:
        return "Good"
    if v >= warn_ge:
        return "Warning"
    return "Poor"


# ---------------------------------------------------------------------
# Dataframe analytics (dataset profiling / quality / correlations)
# ---------------------------------------------------------------------

def dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Row/column/missing/duplicate/memory profile of a dataset."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "n_numeric": int(df.select_dtypes(include=["number"]).shape[1]),
        "n_categorical": int(
            df.select_dtypes(include=["object", "category"]).shape[1]
        ),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "column_names": list(df.columns),
    }


def data_quality_report(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Per-feature missing%/constant/dtype report."""
    total = max(len(df), 1)
    rows = []
    for col in df.columns:
        miss = int(df[col].isna().sum())
        nunique = int(df[col].nunique(dropna=True))
        rows.append({
            "feature": col,
            "dtype": str(df[col].dtype),
            "missing": miss,
            "missing_pct": round(100.0 * miss / total, 2),
            "unique": nunique,
            "constant": nunique <= 1,
        })
    return rows


def invalid_numeric(df: pd.DataFrame) -> Dict[str, int]:
    """Count non-numeric cells in non-numeric-looking columns."""
    out: Dict[str, int] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad = int(coerced.isna().sum() - df[col].isna().sum())
        if bad > 0:
            out[col] = bad
    return out


def describe_target(df: pd.DataFrame, col: str) -> Optional[Dict[str, Any]]:
    """Descriptive statistics for a single (numeric) target column."""
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    count = int(s.count())
    return {
        "count": count,
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0) if count > 1 else 0.0),
        "min": float(s.min()),
        "max": float(s.max()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "missing": int(df[col].isna().sum()),
    }


def numeric_correlation_matrix(df: pd.DataFrame):
    """Correlation matrix of numeric columns (None if <2 numeric cols)."""
    num = df.select_dtypes(include=["number"])
    if num.shape[1] < 2 or num.shape[0] == 0:
        return None
    return num.corr()


def top_correlations(df: pd.DataFrame, target_col: str, top_n: int = 10):
    """Top absolute correlations of a target with other numeric features."""
    num = df.select_dtypes(include=["number"])
    if target_col not in num.columns:
        return []
    corr = num.corr()[target_col].drop(target_col, errors="ignore").dropna()
    return [
        {"feature": k, "correlation": float(v)}
        for k, v in corr.abs().sort_values(ascending=False).head(top_n).items()
    ]
