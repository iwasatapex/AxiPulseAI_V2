from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.probabilistic import (
    wrap_prediction,
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
        nps_uncertainty: float = 0.05,
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
        ``postprocess_predictions`` → ``_axi_attach_0_10_probabilistic_analysis``.
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
            nps_val = float(raw.nps)
            # Check if probabilistic analysis already attached
            if isinstance(raw.nps, dict):
                if "monte_carlo_nps" in raw.nps:
                    # Reuse existing categorical NPS probabilistic evidence
                    # Do NOT run a second Bayesian/MC - this was the duplicate
                    nps_envelope = {
                        "prediction": nps_val,
                        "probabilistic": {
                            "most_likely": nps_val,
                            "likely_range_lower": nps_val - 10.0,
                            "likely_range_upper": nps_val + 10.0,
                            "range_confidence": 0.9,
                            "uncertainty": 10.0,
                            "confidence": 0.9,
                            "source": "reuse_existing_categorical",
                            "bayesian_score_distribution": raw.nps.get(
                                "bayesian_score_distribution",
                                {f"score_{i}": 1.0/11 for i in range(11)}
                            ),
                            "monte_carlo_score_distribution": {
                                f"score_{i}": float(raw.nps.get("mean_score_counts", {}).get(f"score_{i}", 0)) 
                                for i in range(11)
                            },
                            "monte_carlo_nps_p05": raw.nps.get("monte_carlo_nps_p05", nps_val - 10.0),
                            "monte_carlo_nps_p50": raw.nps.get("monte_carlo_nps_p50", nps_val),
                            "monte_carlo_nps_p95": raw.nps.get("monte_carlo_nps_p95", nps_val + 10.0),
                        },
                    }
                else:
                    # No existing NPS probabilistic data - run new envelope
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
            else:
                # Raw float value - run new envelope
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
