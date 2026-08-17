from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    brier_score: float
    mean_predicted_probability: float
    observed_frequency: float
    sample_count: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    predicted_probability: float
    observed_frequency: float
    samples: int


@dataclass(frozen=True)
class ReliabilityResult:
    bins: tuple[ReliabilityBin, ...]
    sample_count: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class IntervalCoverageResult:
    coverage: float
    covered: int
    total: int
    nominal_level: float
    metadata: dict[str, object]


class CalibrationEngine:
    """Universal, model-agnostic calibration metrics."""

    @staticmethod
    def binary(
        probabilities: Iterable[float],
        outcomes: Iterable[float],
    ) -> CalibrationResult:
        probabilities = np.asarray(list(probabilities), dtype=float)
        outcomes = np.asarray(list(outcomes), dtype=float)

        if probabilities.ndim != 1 or outcomes.ndim != 1:
            raise ValueError("probabilities and outcomes must be one-dimensional")

        if len(probabilities) != len(outcomes):
            raise ValueError("probabilities and outcomes must have equal length")

        if len(probabilities) == 0:
            raise ValueError("calibration inputs must not be empty")

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must be finite")

        if not np.all(np.isfinite(outcomes)):
            raise ValueError("outcomes must be finite")

        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("probabilities must be between 0 and 1")

        if np.any((outcomes != 0.0) & (outcomes != 1.0)):
            raise ValueError("outcomes must be binary 0 or 1")

        brier = float(np.mean((probabilities - outcomes) ** 2))

        return CalibrationResult(
            brier_score=brier,
            mean_predicted_probability=float(np.mean(probabilities)),
            observed_frequency=float(np.mean(outcomes)),
            sample_count=len(probabilities),
            metadata={
                "metric": "brier_score",
                "prediction_type": "binary_probability",
            },
        )

    @staticmethod
    def reliability(
        probabilities: Iterable[float],
        outcomes: Iterable[float],
        *,
        bins: int = 10,
    ) -> ReliabilityResult:
        probabilities = np.asarray(list(probabilities), dtype=float)
        outcomes = np.asarray(list(outcomes), dtype=float)

        if len(probabilities) != len(outcomes):
            raise ValueError("probabilities and outcomes must have equal length")

        if len(probabilities) == 0:
            raise ValueError("calibration inputs must not be empty")

        if bins <= 0:
            raise ValueError("bins must be positive")

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must be finite")

        if not np.all(np.isfinite(outcomes)):
            raise ValueError("outcomes must be finite")

        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("probabilities must be between 0 and 1")

        if np.any((outcomes != 0.0) & (outcomes != 1.0)):
            raise ValueError("outcomes must be binary 0 or 1")

        edges = np.linspace(0.0, 1.0, bins + 1)
        results: list[ReliabilityBin] = []

        for index in range(bins):
            lower = float(edges[index])
            upper = float(edges[index + 1])

            if index == bins - 1:
                mask = (probabilities >= lower) & (probabilities <= upper)
            else:
                mask = (probabilities >= lower) & (probabilities < upper)

            count = int(np.sum(mask))

            if count == 0:
                continue

            results.append(
                ReliabilityBin(
                    lower=lower,
                    upper=upper,
                    predicted_probability=float(np.mean(probabilities[mask])),
                    observed_frequency=float(np.mean(outcomes[mask])),
                    samples=count,
                )
            )

        return ReliabilityResult(
            bins=tuple(results),
            sample_count=len(probabilities),
            metadata={
                "metric": "reliability",
                "bin_count": bins,
            },
        )

    @staticmethod
    def interval_coverage(
        lower: Iterable[float],
        upper: Iterable[float],
        actual: Iterable[float],
        *,
        nominal_level: float = 0.95,
    ) -> IntervalCoverageResult:
        lower = np.asarray(list(lower), dtype=float)
        upper = np.asarray(list(upper), dtype=float)
        actual = np.asarray(list(actual), dtype=float)

        if not (
            len(lower) == len(upper) == len(actual)
        ):
            raise ValueError("interval arrays must have equal length")

        if len(actual) == 0:
            raise ValueError("interval inputs must not be empty")

        if not 0.0 < nominal_level < 1.0:
            raise ValueError("nominal_level must be between 0 and 1")

        if not (
            np.all(np.isfinite(lower))
            and np.all(np.isfinite(upper))
            and np.all(np.isfinite(actual))
        ):
            raise ValueError("interval inputs must be finite")

        if np.any(lower > upper):
            raise ValueError("lower bounds must not exceed upper bounds")

        covered = int(np.sum((actual >= lower) & (actual <= upper)))
        total = len(actual)

        return IntervalCoverageResult(
            coverage=float(covered / total),
            covered=covered,
            total=total,
            nominal_level=float(nominal_level),
            metadata={
                "metric": "prediction_interval_coverage",
            },
        )


__all__ = [
    "CalibrationEngine",
    "CalibrationResult",
    "ReliabilityBin",
    "ReliabilityResult",
    "IntervalCoverageResult",
]
