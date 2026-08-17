from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PredictionDriftResult:
    reference_mean: float
    current_mean: float
    mean_shift: float
    reference_std: float
    current_std: float
    std_shift: float
    reference_median: float
    current_median: float
    median_shift: float
    drift_detected: bool
    mean_threshold: float
    std_threshold: float
    median_threshold: float
    metadata: dict[str, object]


def _values(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]

    if not result:
        raise ValueError("predictions must not be empty")

    return result


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    return float(variance ** 0.5)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2

    if len(ordered) % 2:
        return float(ordered[middle])

    return float(
        (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def detect_prediction_drift(
    reference_predictions: Iterable[float],
    current_predictions: Iterable[float],
    *,
    mean_threshold: float = 0.0,
    std_threshold: float = 0.0,
    median_threshold: float = 0.0,
) -> PredictionDriftResult:
    """
    Compare prediction distributions between reference and current data.

    Observational only:
    - does not mutate predictions
    - does not alter predictor behavior
    - does not retrain models
    - does not replace model artifacts
    """
    if mean_threshold < 0.0:
        raise ValueError(
            "mean_threshold must be non-negative"
        )

    if std_threshold < 0.0:
        raise ValueError(
            "std_threshold must be non-negative"
        )

    if median_threshold < 0.0:
        raise ValueError(
            "median_threshold must be non-negative"
        )

    reference = _values(reference_predictions)
    current = _values(current_predictions)

    reference_mean = _mean(reference)
    current_mean = _mean(current)

    reference_std = _std(
        reference,
        reference_mean,
    )
    current_std = _std(
        current,
        current_mean,
    )

    reference_median = _median(reference)
    current_median = _median(current)

    mean_shift = current_mean - reference_mean
    std_shift = current_std - reference_std
    median_shift = current_median - reference_median

    mean_drift = abs(mean_shift) > mean_threshold
    std_drift = abs(std_shift) > std_threshold
    median_drift = abs(median_shift) > median_threshold

    return PredictionDriftResult(
        reference_mean=reference_mean,
        current_mean=current_mean,
        mean_shift=mean_shift,
        reference_std=reference_std,
        current_std=current_std,
        std_shift=std_shift,
        reference_median=reference_median,
        current_median=current_median,
        median_shift=median_shift,
        drift_detected=(
            mean_drift
            or std_drift
            or median_drift
        ),
        mean_threshold=float(mean_threshold),
        std_threshold=float(std_threshold),
        median_threshold=float(median_threshold),
        metadata={
            "detection_mode": "prediction_distribution_comparison",
            "observational": True,
        },
    )
