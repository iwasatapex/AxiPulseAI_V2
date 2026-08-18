from __future__ import annotations

from typing import Any, Mapping

from .pipeline import (
    PredictionPipelineResult,
    ProductionPredictionPipeline,
)


def predict_production(
    state: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    operations_health_target: float | None = None,
    nps_target: float | None = None,
    operations_health_uncertainty: float = 0.05,
    nps_uncertainty: float = 0.05,
    simulations: int = 10000,
    seed: int = 0,
) -> PredictionPipelineResult:
    """
    Public production prediction entrypoint.

    Delegates to the existing production prediction pipeline.
    No predictor or model logic is implemented here.
    """
    pipeline = ProductionPredictionPipeline()

    return pipeline.run(
        state,
        metadata=metadata,
        operations_health_target=operations_health_target,
        nps_target=nps_target,
        operations_health_uncertainty=operations_health_uncertainty,
        nps_uncertainty=nps_uncertainty,
        simulations=simulations,
        seed=seed,
    )


__all__ = ["predict_production"]
