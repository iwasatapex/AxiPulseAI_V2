from __future__ import annotations

from typing import Any

from core.probabilistic import (
    UniversalPredictionEnvelope,
    wrap_prediction,
)


def adapt_nps_prediction(
    prediction: float,
    *,
    target: float | None = None,
    uncertainty: float = 2.0,
    observations: list[float] | None = None,
    samples: int = 10000,
    seed: int = 0,
    metadata: dict[str, Any] | None = None,
) -> UniversalPredictionEnvelope:
    """
    Add universal probabilistic information to an existing NPS prediction.

    The supplied NPS prediction is never modified.
    """

    merged_metadata = {
        "predictor": "nps",
        "metric": "nps",
        **(metadata or {}),
    }

    return wrap_prediction(
        float(prediction),
        target=target,
        uncertainty=uncertainty,
        observations=observations,
        samples=samples,
        seed=seed,
        metadata=merged_metadata,
    )


def adapt_nps_result(
    result: Any,
    *,
    target: float | None = None,
    uncertainty: float = 2.0,
    observations: list[float] | None = None,
    samples: int = 10000,
    seed: int = 0,
) -> UniversalPredictionEnvelope:
    """
    Adapt common existing NPS return forms without changing them.

    Supported:
      - scalar
      - tuple where the NPS value is the second element
      - dict containing 'nps'
    """

    if isinstance(result, tuple):
        if len(result) < 2:
            raise ValueError("NPS tuple result must contain an NPS value")
        prediction = float(result[1])

    elif isinstance(result, dict):
        if "nps" not in result:
            raise ValueError("NPS result dictionary must contain 'nps'")
        prediction = float(result["nps"])

    else:
        prediction = float(result)

    return adapt_nps_prediction(
        prediction,
        target=target,
        uncertainty=uncertainty,
        observations=observations,
        samples=samples,
        seed=seed,
    )
