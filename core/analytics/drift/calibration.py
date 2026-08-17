from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CalibrationDriftResult:
    reference_brier_score: float
    current_brier_score: float
    brier_score_shift: float
    reference_probability_mae: float
    current_probability_mae: float
    probability_mae_shift: float
    drift_detected: bool
    brier_threshold: float
    probability_mae_threshold: float
    metadata: dict[str, object]


def _validate(
    probabilities: Iterable[float],
    outcomes: Iterable[float],
) -> tuple[list[float], list[float]]:
    probs = [float(value) for value in probabilities]
    actuals = [float(value) for value in outcomes]

    if not probs or not actuals:
        raise ValueError("probabilities and outcomes must not be empty")

    if len(probs) != len(actuals):
        raise ValueError(
            "probabilities and outcomes must have equal lengths"
        )

    if any(value < 0.0 or value > 1.0 for value in probs):
        raise ValueError(
            "probabilities must be between 0 and 1"
        )

    if any(value not in (0.0, 1.0) for value in actuals):
        raise ValueError(
            "outcomes must be binary 0 or 1"
        )

    return probs, actuals


def _brier(
    probabilities: list[float],
    outcomes: list[float],
) -> float:
    return float(
        sum(
            (probability - outcome) ** 2
            for probability, outcome
            in zip(probabilities, outcomes)
        )
        / len(probabilities)
    )


def _probability_mae(
    probabilities: list[float],
    outcomes: list[float],
) -> float:
    return float(
        sum(
            abs(probability - outcome)
            for probability, outcome
            in zip(probabilities, outcomes)
        )
        / len(probabilities)
    )


def detect_calibration_drift(
    reference_probabilities: Iterable[float],
    reference_outcomes: Iterable[float],
    current_probabilities: Iterable[float],
    current_outcomes: Iterable[float],
    *,
    brier_threshold: float = 0.0,
    probability_mae_threshold: float = 0.0,
) -> CalibrationDriftResult:
    """
    Compare calibration quality between reference and current periods.

    Observational only:
    - no model mutation
    - no predictor mutation
    - no retraining
    - no automatic deployment decision
    """
    if brier_threshold < 0.0:
        raise ValueError(
            "brier_threshold must be non-negative"
        )

    if probability_mae_threshold < 0.0:
        raise ValueError(
            "probability_mae_threshold must be non-negative"
        )

    ref_probs, ref_actuals = _validate(
        reference_probabilities,
        reference_outcomes,
    )

    cur_probs, cur_actuals = _validate(
        current_probabilities,
        current_outcomes,
    )

    reference_brier = _brier(ref_probs, ref_actuals)
    current_brier = _brier(cur_probs, cur_actuals)

    reference_mae = _probability_mae(
        ref_probs,
        ref_actuals,
    )
    current_mae = _probability_mae(
        cur_probs,
        cur_actuals,
    )

    brier_shift = current_brier - reference_brier
    mae_shift = current_mae - reference_mae

    brier_drift = brier_shift > brier_threshold
    mae_drift = mae_shift > probability_mae_threshold

    return CalibrationDriftResult(
        reference_brier_score=reference_brier,
        current_brier_score=current_brier,
        brier_score_shift=brier_shift,
        reference_probability_mae=reference_mae,
        current_probability_mae=current_mae,
        probability_mae_shift=mae_shift,
        drift_detected=brier_drift or mae_drift,
        brier_threshold=float(brier_threshold),
        probability_mae_threshold=float(
            probability_mae_threshold
        ),
        metadata={
            "detection_mode": "calibration_quality_comparison",
            "observational": True,
        },
    )
