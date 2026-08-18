"""
Universal additive adapters for prediction-producing domains.

This module deliberately does NOT alter predictor calculations.

It converts already-produced scalar/dict/tuple prediction values into
the universal probabilistic result/envelope contract.

Business logic remains owned by the originating predictor.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .prediction_envelope import UniversalPredictionEnvelope, wrap_prediction


def adapt_domain_prediction(
    value: Any,
    *,
    predictor: str,
    metric: str | None = None,
    target: float | None = None,
    uncertainty: float = 0.05,
    simulations: int = 10000,
    seed: int = 0,
) -> UniversalPredictionEnvelope:
    """
    Adapt an existing prediction without changing its value.

    The original prediction is preserved as raw_prediction.
    """

    return wrap_prediction(
        value,
        target=target,
        uncertainty=uncertainty,
        samples=simulations,
        seed=seed,
        metadata={
            "predictor": predictor,
            "metric": metric,
        },
    )


def adapt_target_state_prediction(
    value: Any,
    *,
    metric: str | None = None,
    target: float | None = None,
    uncertainty: float = 0.05,
    simulations: int = 10000,
    seed: int = 0,
) -> UniversalPredictionEnvelope:
    return adapt_domain_prediction(
        value,
        predictor="TargetStateEngine",
        metric=metric,
        target=target,
        uncertainty=uncertainty,
        simulations=simulations,
        seed=seed,
    )


__all__ = [
    "adapt_domain_prediction",
    "adapt_target_state_prediction",
]
