from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class PredictionMetrics:
    count: int
    mae: float
    rmse: float
    bias: float


def calculate_prediction_metrics(
    predicted: Iterable[float],
    actual: Iterable[float],
) -> PredictionMetrics:
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

    return PredictionMetrics(
        count=len(errors),
        mae=float(sum(absolute_errors) / len(errors)),
        rmse=float(sqrt(sum(squared_errors) / len(errors))),
        bias=float(sum(errors) / len(errors)),
    )
