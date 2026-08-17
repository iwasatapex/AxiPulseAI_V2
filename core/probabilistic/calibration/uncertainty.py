from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class UncertaintyValidationResult:
    coverage: float
    nominal_level: float
    covered: int
    total: int
    mean_interval_width: float
    median_interval_width: float
    min_interval_width: float
    max_interval_width: float
    coverage_error: float
    metadata: dict[str, object]


class UncertaintyValidator:
    """Model-agnostic validation of predictive interval quality."""

    @staticmethod
    def validate_intervals(
        lower: Iterable[float],
        upper: Iterable[float],
        actual: Iterable[float],
        *,
        nominal_level: float = 0.95,
    ) -> UncertaintyValidationResult:

        lower = np.asarray(list(lower), dtype=float)
        upper = np.asarray(list(upper), dtype=float)
        actual = np.asarray(list(actual), dtype=float)

        if not (
            len(lower) == len(upper) == len(actual)
        ):
            raise ValueError(
                "lower, upper, and actual must have equal length"
            )

        if len(actual) == 0:
            raise ValueError("validation inputs must not be empty")

        if not 0.0 < nominal_level < 1.0:
            raise ValueError(
                "nominal_level must be between 0 and 1"
            )

        if not (
            np.all(np.isfinite(lower))
            and np.all(np.isfinite(upper))
            and np.all(np.isfinite(actual))
        ):
            raise ValueError(
                "validation inputs must be finite"
            )

        if np.any(lower > upper):
            raise ValueError(
                "lower bounds must not exceed upper bounds"
            )

        widths = upper - lower

        if np.any(widths < 0.0):
            raise ValueError(
                "interval widths must be non-negative"
            )

        covered_mask = (
            (actual >= lower)
            & (actual <= upper)
        )

        covered = int(np.sum(covered_mask))
        total = len(actual)

        coverage = float(covered / total)

        return UncertaintyValidationResult(
            coverage=coverage,
            nominal_level=float(nominal_level),
            covered=covered,
            total=total,
            mean_interval_width=float(np.mean(widths)),
            median_interval_width=float(np.median(widths)),
            min_interval_width=float(np.min(widths)),
            max_interval_width=float(np.max(widths)),
            coverage_error=float(
                abs(coverage - nominal_level)
            ),
            metadata={
                "metric": "prediction_interval_validation",
                "coverage_definition": (
                    "actual between inclusive lower and upper"
                ),
            },
        )


__all__ = [
    "UncertaintyValidator",
    "UncertaintyValidationResult",
]
