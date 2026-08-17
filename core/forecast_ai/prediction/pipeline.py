from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from core.analytics.prediction import PredictionRecord

from .production import (
    ProductionPredictionAdapter,
    ProductionPredictionResult,
)


@dataclass(frozen=True)
class PredictionPipelineResult:
    """
    Canonical production prediction-pipeline response.

    The pipeline is an orchestration boundary only. It does not alter
    predictor calculations or model artifacts.
    """

    prediction: ProductionPredictionResult
    request_metadata: dict[str, Any] = field(default_factory=dict)
    prediction_records: tuple[PredictionRecord, ...] = ()


class ProductionPredictionPipeline:
    """
    Thin orchestration layer for production prediction.

    Responsibilities:
    - accept a prediction state
    - preserve request metadata
    - delegate prediction to the existing production adapter
    - return the existing raw prediction plus additive probabilistic data

    It does not:
    - retrain models
    - replace models
    - modify predictor formulas
    - mutate caller input
    - generate outcomes
    """

    def __init__(
        self,
        adapter: ProductionPredictionAdapter | None = None,
    ) -> None:
        self.adapter = adapter or ProductionPredictionAdapter()
        self.prediction_records: list[PredictionRecord] = []

    def run(
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
    ) -> PredictionPipelineResult:
        request_metadata = dict(metadata or {})

        prediction = self.adapter.predict(
            dict(state),
            metadata=request_metadata,
            operations_health_target=operations_health_target,
            nps_target=nps_target,
            operations_health_uncertainty=operations_health_uncertainty,
            nps_uncertainty=nps_uncertainty,
            simulations=simulations,
            seed=seed,
        )

        # Observational only: retain a prediction record.
        # No actual outcome is fabricated at prediction time.
        #
        # The production result remains untouched; this record is an
        # additive analytics artifact only.
        if isinstance(prediction.raw, (int, float)):
            self.prediction_records.append(
                PredictionRecord(
                    prediction_id=str(
                        request_metadata.get(
                            "prediction_id",
                            len(self.prediction_records) + 1,
                        )
                    ),
                    predicted=float(prediction.raw),
                    model_version=request_metadata.get("model_version"),
                    dataset_version=request_metadata.get("dataset_version"),
                    feature_version=request_metadata.get("feature_version"),
                    metadata=request_metadata,
                )
            )

        return PredictionPipelineResult(
            prediction=prediction,
            request_metadata=request_metadata,
            prediction_records=tuple(self.prediction_records),
        )


__all__ = [
    "PredictionPipelineResult",
    "ProductionPredictionPipeline",
]
