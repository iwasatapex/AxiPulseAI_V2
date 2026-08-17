from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class DriftMetrics:
    count_reference: int
    count_current: int
    reference_mean: float
    current_mean: float
    mean_shift: float
    mean_absolute_shift: float
    reference_std: float
    current_std: float
    std_shift: float


def _values(values: Iterable[float], name: str) -> list[float]:
    result = [float(value) for value in values]

    if not result:
        raise ValueError(f"{name} must not be empty")

    if any(not isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite numeric values")

    return result


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _population_std(values: list[float], mean: float) -> float:
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return float(variance ** 0.5)


def calculate_numeric_drift(
    reference: Iterable[float],
    current: Iterable[float],
) -> DriftMetrics:
    """
    Calculate descriptive numeric drift metrics.

    Observational only:
    - does not mutate inputs
    - does not retrain models
    - does not alter predictor behavior
    - does not make deployment decisions
    """
    reference_values = _values(reference, "reference")
    current_values = _values(current, "current")

    reference_mean = _mean(reference_values)
    current_mean = _mean(current_values)

    reference_std = _population_std(
        reference_values,
        reference_mean,
    )
    current_std = _population_std(
        current_values,
        current_mean,
    )

    mean_shift = current_mean - reference_mean
    std_shift = current_std - reference_std

    return DriftMetrics(
        count_reference=len(reference_values),
        count_current=len(current_values),
        reference_mean=reference_mean,
        current_mean=current_mean,
        mean_shift=mean_shift,
        mean_absolute_shift=abs(mean_shift),
        reference_std=reference_std,
        current_std=current_std,
        std_shift=std_shift,
    )
