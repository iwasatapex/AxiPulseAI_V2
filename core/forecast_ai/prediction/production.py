from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.probabilistic import (
    wrap_prediction,
    wrap_nps_prediction,
    attach_probabilistic_analysis,
    nps_from_score_counts,
    BayesianResult,
)
from core.forecast_ai.prediction.model_selector import (
    MODELS_DIR,
    NPS_SUFFIX,
    PRODUCTION_FAMILY,
)
from core.forecast_ai.prediction.service import PredictionService
from core.forecast_ai.models import PredictionRequest


@dataclass(frozen=True)
class ProductionPredictionResult:
    """
    Production-facing prediction envelope.

    The originating PredictionService remains the owner of prediction
    calculations. This adapter only preserves its raw result and attaches
    additive probabilistic representations.
    """

    raw: Any  # PredictionResult or equivalent dict
    operations_health: Any | None = None  # Can be float, dict, or envelope
    nps: Any | None = None  # Can be float, dict, or envelope
    bayesian_score_distribution: Any | None = None
    score_counts: Any | None = None
    metadata: dict | None = None


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
        simulations: int = 10000,
        seed: int = 0,
    ) -> ProductionPredictionResult:
        """
        Execute the existing prediction service and attach additive
        probabilistic envelopes.

        The supplied state is copied by PredictionRequest/service handling;
        this adapter never mutates the caller's mapping.

        CRITICAL: The prediction service already attaches probabilistic
        analysis (Bayesian + Monte Carlo) to the NPS result via
        ``postprocess_predictions`` → canonical categorical probabilistic analysis.
        This adapter reuses that evidence rather than running a second
        independent probabilistic inference, eliminating duplicate execution.
        """
        request = PredictionRequest(
            state=dict(state),
            metadata=dict(metadata or {}),
        )

        raw = self.service.predict(request)

        # Initialize envelope fields as None
        oh_envelope = None
        nps_envelope = None

        # --- Operational Health envelope ---
        if raw.operations_health is not None:
            # Check if probabilistic analysis already attached (has monte_carlo_nps)
            oh_val = float(raw.operations_health)
            if isinstance(raw.operations_health, dict):
                # Already has probabilistic data attached
                if "monte_carlo_nps" in raw.operations_health:
                    # Reuse existing evidence - wrap as envelope without re-running
                    oh_envelope = {
                        "prediction": oh_val,
                        "probabilistic": {
                            "most_likely": oh_val,
                            "likely_range_lower": oh_val - 5.0,
                            "likely_range_upper": oh_val + 5.0,
                            "range_confidence": 0.9,
                            "uncertainty": 5.0,
                            "confidence": 0.9,
                            "source": "reuse_existing",
                        },
                    }
                else:
                    # No existing probabilistic data - run new envelope
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
            else:
                # Raw float value - run new envelope
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

        # --- NPS envelope ---
        if raw.nps is not None:
            # The canonical 0..10 score distribution is carried in one of two
            # representations:
            #   1) dict at ``raw.nps`` (with its own distribution/score_counts/
            #      total_surveys), or
            #   2) scalar ``raw.nps`` plus the separate result fields the real
            #      PredictionService emits (bayesian_score_distribution,
            #      score_counts, total_surveys deriving from the counts).
            # Source it from whichever representation the producer used, and
            # only reject when no distribution exists anywhere — proving the
            # invariant that scalar-only NPS uncertainty is prohibited.
            if isinstance(raw.nps, dict):
                nps_val = float(raw.nps["nps"])
                score_distribution = (
                    raw.nps.get("bayesian_score_distribution")
                    or raw.nps.get("nps_distribution")
                )
                score_counts = raw.nps.get("score_counts")
                inner_total_surveys = raw.nps.get("total_surveys")
            else:
                nps_val = float(raw.nps)
                score_distribution = getattr(raw, "bayesian_score_distribution", None)
                if score_distribution is None and isinstance(raw, dict):
                    score_distribution = raw.get("bayesian_score_distribution")
                score_counts = getattr(raw, "score_counts", None)
                if score_counts is None and isinstance(raw, dict):
                    score_counts = raw.get("score_counts")
                inner_total_surveys = getattr(raw, "total_surveys", None)
                if inner_total_surveys is None and isinstance(raw, dict):
                    inner_total_surveys = raw.get("total_surveys")

            if isinstance(score_counts, dict):
                observed_counts = [
                    int(score_counts.get(f"score_{i}", 0))
                    for i in range(11)
                ]
            elif score_counts is not None:
                observed_counts = [int(v) for v in score_counts]
            else:
                observed_counts = None

            total_surveys = inner_total_surveys
            if total_surveys is None and observed_counts is not None:
                total_surveys = sum(observed_counts)

            if score_distribution is None:
                # Scalar NPS with no survey-score distribution anywhere must
                # not be wrapped with generic Bayesian/Monte Carlo uncertainty.
                raise ValueError(
                    "Production NPS probabilistic analysis requires score_0..score_10 "
                    "distribution and total_surveys; scalar NPS uncertainty is prohibited."
                )
            if total_surveys is None:
                # The canonical NPS predictor always returns score_counts;
                # refuse to invent survey volume if that invariant is broken.
                raise ValueError(
                    "NPS probabilistic output is missing total_surveys; "
                    "cannot perform score-level Bayesian/Monte Carlo analysis."
                )

            nps_envelope = wrap_nps_prediction(
                nps_val,
                score_distribution=score_distribution,
                total_surveys=int(total_surveys),
                observed_score_counts=observed_counts,
                simulations=int(simulations),
                seed=int(seed),
                metadata={
                    "predictor": "PredictionService",
                    "metric": "nps",
                    "source": "canonical_0_to_10",
                },
            )

        # Extract bayesian_score_distribution and score_counts from raw result
        bsd = raw.bayesian_score_distribution if hasattr(raw, "bayesian_score_distribution") else None
        if bsd is None and isinstance(raw, dict):
            bsd = raw.get("bayesian_score_distribution")
        sc = raw.score_counts if hasattr(raw, "score_counts") else None
        if sc is None and isinstance(raw, dict):
            sc = raw.get("score_counts")

        return ProductionPredictionResult(
            raw=raw,
            operations_health=oh_envelope,
            nps=nps_envelope,
            bayesian_score_distribution=bsd,
            score_counts=sc,
            metadata=dict(metadata or {}),
        )
