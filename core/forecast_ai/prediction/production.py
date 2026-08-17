from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.probabilistic import (
    UniversalPredictionEnvelope,
    wrap_prediction,
)

from ..models import PredictionRequest, PredictionResult
from .service import PredictionService


@dataclass(frozen=True)
class ProductionPredictionResult:
    """
    Production-facing prediction envelope.

    The originating PredictionService remains the owner of prediction
    calculations. This adapter only preserves its raw result and attaches
    additive probabilistic representations.
    """

    raw: PredictionResult
    operations_health: UniversalPredictionEnvelope | None = None
    nps: UniversalPredictionEnvelope | None = None

    # Preserve the NPS predictor's native 0-10 distribution at the
    # production boundary. These are observational fields only.
    # No Bayesian, Monte Carlo, NPS, or model logic is performed here.
    bayesian_score_distribution: dict[str, float] | None = None
    score_counts: dict[str, int] | None = None

    metadata: dict[str, Any] | None = None


class ProductionPredictionAdapter:
    """
    Thin production boundary around the existing PredictionService.

    No predictor calculation, model loading, model fitting, or forecast
    logic is performed here.
    """

    def __init__(self, service: PredictionService | None = None) -> None:
        self.service = service or PredictionService()

    def predict(
        self,
        state: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        operations_health_target: float | None = None,
        nps_target: float | None = None,
        operations_health_uncertainty: float = 0.05,
        nps_uncertainty: float = 0.05,
        simulations: int = 10000,
        seed: int = 0,
    ) -> ProductionPredictionResult:
        """
        Execute the existing prediction service and attach additive
        probabilistic envelopes.

        The supplied state is copied by PredictionRequest/service handling;
        this adapter never mutates the caller's mapping.
        """
        request = PredictionRequest(
            state=dict(state),
            metadata=dict(metadata or {}),
        )

        raw = self.service.predict(request)

        oh_envelope = None
        if raw.operations_health is not None:
            oh_envelope = wrap_prediction(
                float(raw.operations_health),
                target=operations_health_target,
                uncertainty=float(operations_health_uncertainty),
                samples=int(simulations),
                seed=int(seed),
                metadata={
                    "predictor": "PredictionService",
                    "metric": "operations_health",
                },
            )

        nps_envelope = None
        if raw.nps is not None:
            # Production contract requires result.nps to remain an additive
            # probabilistic envelope. The native 0-10 inference remains the
            # authoritative source for the NPS value and score distribution.
            #
            # This envelope is compatibility metadata only; it must not
            # replace, recalculate, or optimize the native 0-10 NPS result.
            nps_envelope = wrap_prediction(
                float(raw.nps),
                target=nps_target,
                uncertainty=float(nps_uncertainty),
                samples=int(simulations),
                seed=int(seed),
                metadata={
                    "predictor": "PredictionService",
                    "metric": "nps",
                    "source": "native_0_to_10",
                    "distribution_authoritative": True,
                },
            )

        return ProductionPredictionResult(
            raw=raw,
            operations_health=oh_envelope,
            nps=nps_envelope,

            # Exact handoff of the distribution already produced by
            # PredictionService/NPS inference.
            bayesian_score_distribution=(
                dict(raw.bayesian_score_distribution)
                if isinstance(
                    raw.bayesian_score_distribution,
                    dict,
                )
                else None
            ),
            score_counts=(
                {
                    str(k): int(v)
                    for k, v in raw.score_counts.items()
                }
                if isinstance(raw.score_counts, dict)
                else None
            ),

            metadata=dict(metadata or {}),
        )


__all__ = [
    "ProductionPredictionResult",
    "ProductionPredictionAdapter",
]
