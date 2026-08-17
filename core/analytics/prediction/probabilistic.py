from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ProbabilisticAnalytics:
    count: int
    brier_score: float
    probability_mae: float
    interval_coverage: float
    mean_interval_width: float


def calculate_probabilistic_analytics(
    probabilities: Iterable[float],
    outcomes: Iterable[float],
    lower_bounds: Iterable[float] | None = None,
    upper_bounds: Iterable[float] | None = None,
) -> ProbabilisticAnalytics:
    probability_values = [float(value) for value in probabilities]
    outcome_values = [float(value) for value in outcomes]

    if len(probability_values) != len(outcome_values):
        raise ValueError(
            "probabilities and outcomes must have equal lengths"
        )

    if not probability_values:
        raise ValueError("probabilistic data must not be empty")

    if any(
        probability < 0.0 or probability > 1.0
        for probability in probability_values
    ):
        raise ValueError("probabilities must be between 0 and 1")

    if any(
        outcome not in (0.0, 1.0)
        for outcome in outcome_values
    ):
        raise ValueError("outcomes must be binary 0 or 1")

    count = len(probability_values)

    squared_probability_errors = [
        (probability - outcome) ** 2
        for probability, outcome
        in zip(probability_values, outcome_values)
    ]

    absolute_probability_errors = [
        abs(probability - outcome)
        for probability, outcome
        in zip(probability_values, outcome_values)
    ]

    interval_coverage = 0.0
    mean_interval_width = 0.0

    if lower_bounds is not None or upper_bounds is not None:
        if lower_bounds is None or upper_bounds is None:
            raise ValueError(
                "lower_bounds and upper_bounds must be supplied together"
            )

        lower_values = [float(value) for value in lower_bounds]
        upper_values = [float(value) for value in upper_bounds]

        if len(lower_values) != count or len(upper_values) != count:
            raise ValueError(
                "interval bounds must match probability length"
            )

        if any(
            lower > upper
            for lower, upper
            in zip(lower_values, upper_values)
        ):
            raise ValueError(
                "interval lower bounds must not exceed upper bounds"
            )

        covered = sum(
            lower <= outcome <= upper
            for lower, upper, outcome
            in zip(
                lower_values,
                upper_values,
                outcome_values,
            )
        )

        widths = [
            upper - lower
            for lower, upper
            in zip(lower_values, upper_values)
        ]

        interval_coverage = float(covered / count)
        mean_interval_width = float(sum(widths) / count)

    return ProbabilisticAnalytics(
        count=count,
        brier_score=float(
            sum(squared_probability_errors) / count
        ),
        probability_mae=float(
            sum(absolute_probability_errors) / count
        ),
        interval_coverage=interval_coverage,
        mean_interval_width=mean_interval_width,
    )
