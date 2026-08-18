from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapter import UniversalProbabilisticAdapter
from .result import ProbabilisticResult


@dataclass(frozen=True)
class UniversalPredictionEnvelope:
    """
    Additive wrapper for an existing scalar prediction.

    The original prediction is preserved exactly.
    Probabilistic information is attached separately.
    """

    prediction: Any
    probabilistic: ProbabilisticResult
    metadata: dict[str, Any]


def wrap_prediction(
    prediction: float,
    *,
    target: float | None = None,
    uncertainty: float = 0.05,
    observations: list[float] | None = None,
    samples: int = 10000,
    seed: int = 0,
    metadata: dict[str, Any] | None = None,
) -> UniversalPredictionEnvelope:
    """
    Convert an existing scalar prediction into the universal
    probabilistic representation without changing the prediction.
    """

    adapter = UniversalProbabilisticAdapter()

    result = adapter.infer(
        observations=observations or [],
        baseline=float(prediction),
        target=target,
        uncertainty=float(uncertainty),
        samples=int(samples),
        seed=int(seed),
    )

    return UniversalPredictionEnvelope(
        prediction=prediction,
        probabilistic=result,
        metadata=dict(metadata or {}),
    )
