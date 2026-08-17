from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.forecast_ai.prediction.production import ProductionPredictionResult

from .probabilistic_decision import (
    ProbabilisticDecisionPackage,
    ProbabilisticDecisionService,
)


@dataclass(frozen=True)
class ProductionDecisionInput:
    """Decision-ready observational representation of a production prediction."""

    metric: str
    prediction: float
    probability: float | None
    confidence: float | None
    expected: float | None
    p05: float | None
    p95: float | None
    metadata: dict[str, Any]


class ProductionDecisionBoundary:
    """
    Additive boundary between the production prediction pipeline and ADIE.

    This boundary does not calculate predictions, alter model outputs, mutate
    caller state, retrain models, or replace production model artifacts.

    Since V3 consolidation, this boundary is an EXPORTED, ENFORCED gate: the
    canonical V3 service calls ``validate()`` before any decision is produced
    for production inputs. It rejects malformed inputs, non-finite values,
    future-dated observed provenance, and predicted-recursive state.
    """

    def __init__(
        self,
        decision_service: ProbabilisticDecisionService | None = None,
    ) -> None:
        self.decision_service = (
            decision_service or ProbabilisticDecisionService()
        )

    # ------------------------------------------------------------------
    # Enforced input validation (production gate)
    # ------------------------------------------------------------------

    @staticmethod
    def validate(
        observations: Sequence[float],
        baseline: float,
        *,
        scenarios: Sequence[Mapping[str, Any]] | None = None,
        cutoff: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Enforce the production decision boundary on decision inputs.

        This is advisory/safety validation only. It does not modify
        predictions, model outputs, or caller state.

        Raises ``ValueError`` (or ``TypeError``) on any violation.
        """
        import math

        from core.common.temporal_contract import assert_known_at_cutoff

        # 1. Scenarios must not be empty when provided.
        if scenarios is not None and not scenarios:
            raise ValueError("scenarios must not be empty")

        # 2. Observations must be non-empty and finite.
        if observations is None or len(observations) == 0:
            raise ValueError("observations must not be empty")
        for item in observations:
            if not math.isfinite(float(item)):
                raise ValueError("observations must contain only finite values")

        # 3. Baseline must be finite.
        if not math.isfinite(float(baseline)):
            raise ValueError("baseline must be a finite number")

        # 4. Observed-vs-predicted separation: a decision input marked as
        #    predicted recursive state (forecast-generated) must NEVER be
        #    treated as observed/realized history.
        md = dict(metadata or {})
        if md.get("is_predicted") and md.get("treated_as_observed"):
            raise ValueError(
                "predicted recursive state must not be treated as observed history"
            )

        # 5. Temporal provenance gate (only when a cutoff is supplied).
        #    Reject any input that is timestamped after the prediction cutoff,
        #    i.e. a future observed outcome cannot enter T inputs.
        if cutoff is not None:
            provenance = md.get("provenance")
            if provenance is None:
                # No provenance supplied: we cannot verify, so we refuse to
                # silently pass future data through. This is the safe default.
                raise ValueError(
                    "temporal provenance required when a cutoff is supplied; "
                    "refusing to accept un-verified decision inputs"
                )
            if isinstance(provenance, (list, tuple)):
                for stamp in provenance:
                    assert_known_at_cutoff(stamp, cutoff, field_name="input")
            else:
                assert_known_at_cutoff(provenance, cutoff, field_name="input")

    @staticmethod
    def _prediction_input(
        metric: str,
        prediction: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProductionDecisionInput:
        value = float(prediction.prediction)

        probabilistic = prediction.probabilistic

        probability = (
            float(probabilistic.probability)
            if probabilistic is not None
            and getattr(probabilistic, "probability", None) is not None
            else None
        )

        confidence = (
            float(probabilistic.confidence)
            if probabilistic is not None
            and getattr(probabilistic, "confidence", None) is not None
            else None
        )

        expected = (
            float(probabilistic.mean)
            if probabilistic is not None
            and getattr(probabilistic, "mean", None) is not None
            else value
        )

        p05 = (
            float(probabilistic.p05)
            if probabilistic is not None
            and getattr(probabilistic, "p05", None) is not None
            else value
        )

        p95 = (
            float(probabilistic.p95)
            if probabilistic is not None
            and getattr(probabilistic, "p95", None) is not None
            else value
        )

        return ProductionDecisionInput(
            metric=metric,
            prediction=value,
            probability=probability,
            confidence=confidence,
            expected=expected,
            p05=p05,
            p95=p95,
            metadata=dict(metadata or {}),
        )

    def prepare(
        self,
        result: ProductionPredictionResult,
    ) -> tuple[ProductionDecisionInput, ...]:
        """Map available production outputs without changing them."""

        if not isinstance(result, ProductionPredictionResult):
            raise TypeError("result must be a ProductionPredictionResult")

        items: list[ProductionDecisionInput] = []

        if result.operations_health is not None:
            items.append(
                self._prediction_input(
                    "operations_health",
                    result.operations_health,
                    metadata=result.metadata,
                )
            )

        if result.nps is not None:
            items.append(
                self._prediction_input(
                    "nps",
                    result.nps,
                    metadata=result.metadata,
                )
            )

        return tuple(items)

    def analyze(
        self,
        result: ProductionPredictionResult,
        scenarios: Sequence[Mapping[str, Any]],
        observations: Sequence[float],
        baseline: float,
        *,
        uncertainty: float = 0.05,
        samples: int = 10000,
    ) -> ProbabilisticDecisionPackage:
        """
        Pass production prediction context into the existing decision service.

        Decision semantics remain owned by ProbabilisticDecisionService.
        """

        prepared = self.prepare(result)

        if not prepared:
            raise ValueError("production prediction contains no decision inputs")

        copied_scenarios = [dict(item) for item in scenarios]

        return self.decision_service.analyze(
            scenarios=copied_scenarios,
            observations=[float(value) for value in observations],
            baseline=float(baseline),
            uncertainty=float(uncertainty),
            samples=int(samples),
        )


__all__ = [
    "ProductionDecisionInput",
    "ProductionDecisionBoundary",
]


@dataclass(frozen=True)
class ProductionOutcomeHandoff:
    """Observational outcome attached after a production prediction."""

    prediction_id: str
    actual_outcome: float
    metadata: dict[str, Any]


class ProductionOutcomeService:
    """
    Attach externally supplied outcomes to prediction records.

    This does not generate outcomes, modify predictions, retrain models,
    or calculate prediction error at prediction time.
    """

    @staticmethod
    def attach(
        prediction_id: str,
        actual_outcome: float,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProductionOutcomeHandoff:
        if not prediction_id:
            raise ValueError("prediction_id must not be empty")

        return ProductionOutcomeHandoff(
            prediction_id=str(prediction_id),
            actual_outcome=float(actual_outcome),
            metadata=dict(metadata or {}),
        )


    @staticmethod
    def calculate_outcome_error(
        prediction: float,
        actual_outcome: float,
    ) -> float:
        """Calculate error only after an actual outcome is supplied."""
        return float(actual_outcome) - float(prediction)


@dataclass(frozen=True)
class OutcomeAnalyticsHandoff:
    """Analytics-ready data produced only after actual outcomes exist."""

    prediction_id: str
    prediction: float
    actual_outcome: float
    error: float
    metadata: dict[str, Any]


class OutcomeAnalyticsService:
    """Bridge external outcomes into the existing observational analytics layer."""

    @staticmethod
    def create(
        prediction_id: str,
        prediction: float,
        actual_outcome: float,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> OutcomeAnalyticsHandoff:
        if not prediction_id:
            raise ValueError("prediction_id must not be empty")

        prediction_value = float(prediction)
        actual_value = float(actual_outcome)

        return OutcomeAnalyticsHandoff(
            prediction_id=str(prediction_id),
            prediction=prediction_value,
            actual_outcome=actual_value,
            error=actual_value - prediction_value,
            metadata=dict(metadata or {}),
        )


    @staticmethod
    def analyze_outcome(
        predictions: list[float],
        outcomes: list[float],
    ) -> dict[str, float]:
        """
        Produce a minimal observational outcome summary.

        Actual outcomes must already exist. This method does not generate
        outcomes and does not modify prediction records or model artifacts.
        """
        if not predictions or not outcomes:
            raise ValueError("predictions and outcomes must not be empty")

        if len(predictions) != len(outcomes):
            raise ValueError("predictions and outcomes must have equal length")

        errors = [
            float(actual) - float(predicted)
            for predicted, actual in zip(predictions, outcomes)
        ]

        absolute_errors = [abs(error) for error in errors]

        return {
            "count": float(len(errors)),
            "mean_error": sum(errors) / len(errors),
            "mean_absolute_error": (
                sum(absolute_errors) / len(absolute_errors)
            ),
        }
