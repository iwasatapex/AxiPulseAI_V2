from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ErrorBiasMetrics:
    count: int
    mean_error: float
    mean_absolute_error: float
    rmse: float
    max_absolute_error: float
    underprediction_rate: float
    overprediction_rate: float


def calculate_error_bias(
    predicted: Iterable[float],
    actual: Iterable[float],
) -> ErrorBiasMetrics:
    predicted_values = [float(value) for value in predicted]
    actual_values = [float(value) for value in actual]

    if len(predicted_values) != len(actual_values):
        raise ValueError("predicted and actual must have equal lengths")

    if not predicted_values:
        raise ValueError("prediction data must not be empty")

    errors = [
        prediction - outcome
        for prediction, outcome
        in zip(predicted_values, actual_values)
    ]

    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]

    count = len(errors)

    return ErrorBiasMetrics(
        count=count,
        mean_error=float(sum(errors) / count),
        mean_absolute_error=float(sum(absolute_errors) / count),
        rmse=float((sum(squared_errors) / count) ** 0.5),
        max_absolute_error=float(max(absolute_errors)),
        underprediction_rate=float(
            sum(error < 0 for error in errors) / count
        ),
        overprediction_rate=float(
            sum(error > 0 for error in errors) / count
        ),
    )
