from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredictionRecord:
    """
    Immutable record connecting a prediction to its eventual outcome.

    This layer is observational only. It does not modify predictors,
    models, probabilistic engines, or calibration state.
    """

    prediction_id: str
    predicted: float
    actual: float | None = None

    model_version: str | None = None
    dataset_version: str | None = None
    feature_version: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def error(self) -> float | None:
        if self.actual is None:
            return None

        return float(self.predicted - self.actual)

    @property
    def absolute_error(self) -> float | None:
        error = self.error

        if error is None:
            return None

        return abs(error)

    @property
    def squared_error(self) -> float | None:
        error = self.error

        if error is None:
            return None

        return float(error * error)

    @property
    def resolved(self) -> bool:
        return self.actual is not None
