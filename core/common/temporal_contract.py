"""Temporal safety contracts for AxiPulseAI forecasting.

Forecasting contract:

    information available at T -> target at T+1

This module performs validation only.
It does not modify model predictions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _coerce_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value

    if value is None:
        raise ValueError("timestamp is required")

    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except Exception as exc:
        raise ValueError(
            f"invalid timestamp: {value!r}"
        ) from exc


def assert_forecast_boundary(
    prediction_cutoff: Any,
    target_time: Any,
) -> None:
    """Require target_time to be strictly after prediction_cutoff."""

    cutoff = _coerce_time(prediction_cutoff)
    target = _coerce_time(target_time)

    if target <= cutoff:
        raise ValueError(
            "Temporal contract violation: "
            "target_time must be strictly after "
            "prediction_cutoff"
        )


def assert_known_at_cutoff(
    feature_time: Any,
    prediction_cutoff: Any,
    *,
    field_name: str = "feature",
) -> None:
    """Reject realized information occurring after cutoff."""

    feature = _coerce_time(feature_time)
    cutoff = _coerce_time(prediction_cutoff)

    if feature > cutoff:
        raise ValueError(
            f"Temporal leakage: {field_name!r} has timestamp "
            f"{feature.isoformat()} after prediction cutoff "
            f"{cutoff.isoformat()}"
        )
