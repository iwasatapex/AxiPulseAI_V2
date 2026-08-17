from __future__ import annotations

from typing import Any

from core.probabilistic import (
    UniversalPredictionEnvelope,
    wrap_prediction,
)


def adapt_oh_prediction(
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
    Add universal probabilistic information to an existing
    Operations Health prediction.

    The supplied OH prediction is never modified.
    """

    merged_metadata = {
        "predictor": "operations_health",
        "metric": "operations_health",
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


def adapt_oh_result(
    result: Any,
    *,
    target: float | None = None,
    uncertainty: float = 2.0,
    observations: list[float] | None = None,
    samples: int = 10000,
    seed: int = 0,
) -> UniversalPredictionEnvelope:
    """
    Adapt common existing OH return forms without changing them.

    Supported:
      - scalar
      - tuple where the prediction is the first element
      - dict containing operations_health / operational_health / prediction
    """

    if isinstance(result, tuple):
        if not result:
            raise ValueError("OH tuple result must contain a prediction")
        prediction = float(result[0])

    elif isinstance(result, dict):
        value = result.get(
            "operations_health",
            result.get(
                "operational_health",
                result.get("prediction"),
            ),
        )

        if value is None:
            raise ValueError(
                "OH result dictionary must contain "
                "'operations_health', 'operational_health', or 'prediction'"
            )

        prediction = float(value)

    else:
        prediction = float(result)

    return adapt_oh_prediction(
        prediction,
        target=target,
        uncertainty=uncertainty,
        observations=observations,
        samples=samples,
        seed=seed,
    )
