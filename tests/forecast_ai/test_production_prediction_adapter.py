from __future__ import annotations

import pytest

from core.forecast_ai.models import PredictionResult
from core.forecast_ai.prediction.production import (
    ProductionPredictionAdapter,
    ProductionPredictionResult,
)
from core.probabilistic import attach_probabilistic_analysis


class StubPredictionService:
    def __init__(self) -> None:
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        # Mirror the real PredictionService: a scalar NPS point forecast plus
        # the canonical 0..10 score distribution carried in the separate
        # result fields (bayesian_score_distribution / score_counts).
        return PredictionResult(
            operations_health=82.0,
            nps=81.0,
            bayesian_score_distribution={
                f"score_{i}": (0.0 if i != 10 else 1.0) for i in range(11)
            },
            score_counts={
                f"score_{i}": (0 if i != 10 else 100) for i in range(11)
            },
            warnings=[],
            errors=[],
        )


def test_adapter_preserves_raw_prediction():
    service = StubPredictionService()
    adapter = ProductionPredictionAdapter(service)

    state = {
        "quality": 87.0,
        "competency": 93.0,
        "release": 60.0,
        "transfer": 9.0,
    }

    result = adapter.predict(
        state,
        metadata={"request_id": "test-1"},
        operations_health_target=80.0,
        nps_target=82.0,
        operations_health_uncertainty=2.0,
        simulations=500,
        seed=42,
    )

    assert isinstance(result, ProductionPredictionResult)
    assert result.raw.operations_health == 82.0
    assert result.raw.nps == 81.0


def test_probabilistic_outputs_are_additive():
    service = StubPredictionService()
    adapter = ProductionPredictionAdapter(service)

    result = adapter.predict(
        {
            "quality": 87.0,
            "competency": 93.0,
            "release": 60.0,
            "transfer": 9.0,
        },
        simulations=500,
        seed=42,
    )

    assert result.operations_health is not None
    assert result.nps is not None

    assert result.operations_health.prediction == 82.0
    assert result.nps.prediction == 81.0

    assert result.operations_health.probabilistic is not None
    assert result.nps.probabilistic is not None


def test_input_state_is_not_mutated():
    service = StubPredictionService()
    adapter = ProductionPredictionAdapter(service)

    state = {
        "quality": 87.0,
        "competency": 93.0,
        "release": 60.0,
        "transfer": 9.0,
    }

    before = dict(state)

    adapter.predict(
        state,
        simulations=100,
        seed=1,
    )

    assert state == before


def test_metadata_is_preserved():
    service = StubPredictionService()
    adapter = ProductionPredictionAdapter(service)

    metadata = {
        "request_id": "phase11",
        "source": "production",
    }

    result = adapter.predict(
        {
            "quality": 87.0,
            "competency": 93.0,
            "release": 60.0,
            "transfer": 9.0,
        },
        metadata=metadata,
        simulations=100,
        seed=1,
    )

    assert result.metadata == metadata
    assert service.requests[0].metadata == metadata


def test_partial_prediction_is_supported():
    class PartialService:
        def predict(self, request):
            return PredictionResult(
                operations_health=82.0,
                nps=None,
                warnings=[],
                errors=["NPS unavailable"],
            )

    adapter = ProductionPredictionAdapter(PartialService())

    result = adapter.predict(
        {"quality": 87.0},
        simulations=100,
        seed=1,
    )

    assert result.operations_health is not None
    assert result.nps is None
    assert result.raw.nps is None
    assert result.raw.errors == ["NPS unavailable"]


# --------------------------------------------------------------------------- #
# NPS uncertainty must originate from the 0..10 survey-score distribution —
# NOT from a scalar NPS ± confidence calculation.
# --------------------------------------------------------------------------- #

# A spread-out 0..10 distribution (promoters/passives/detractors all present)
# so Monte Carlo P05 != P95, making the assertions meaningful.
_MIXED_DIST = {f"score_{i}": 0.0 for i in range(11)}
_MIXED_DIST["score_6"] = 0.05
_MIXED_DIST["score_7"] = 0.15
_MIXED_DIST["score_8"] = 0.30
_MIXED_DIST["score_9"] = 0.35
_MIXED_DIST["score_10"] = 0.15
_MIXED_COUNTS = {
    f"score_{i}": int(round(_MIXED_DIST[f"score_{i}"] * 200))
    for i in range(11)
}  # sums to 200 -> total_surveys = 200


class ScalarNpsService:
    """Returns a scalar NPS point forecast plus the canonical distribution."""

    def __init__(self, nps: float) -> None:
        self.nps = nps

    def predict(self, request):
        return PredictionResult(
            operations_health=82.0,
            nps=self.nps,
            bayesian_score_distribution=dict(_MIXED_DIST),
            score_counts=dict(_MIXED_COUNTS),
            warnings=[],
            errors=[],
        )


def test_nps_interval_comes_from_monte_carlo_nps_percentiles():
    adapter = ProductionPredictionAdapter(ScalarNpsService(50.0))

    result = adapter.predict({}, simulations=1000, seed=42)

    prob = result.nps.probabilistic
    assert prob is not None

    # The canonical producer's own single Monte Carlo run (same distribution,
    # survey volume, simulations and seed) yields the authoritative P05/P95.
    canonical = attach_probabilistic_analysis(
        {"nps": 50.0, "bayesian_score_distribution": dict(_MIXED_DIST)},
        total_surveys=sum(_MIXED_COUNTS.values()),
        observed_counts=[_MIXED_COUNTS[f"score_{i}"] for i in range(11)],
        simulations=1000,
        seed=42,
    )

    assert prob.likely_range_lower == pytest.approx(
        canonical["monte_carlo_nps_p05"], abs=1e-6
    )
    assert prob.likely_range_upper == pytest.approx(
        canonical["monte_carlo_nps_p95"], abs=1e-6
    )
    # The interval is real uncertainty (P05 < P95), not a degenerate scalar band.
    assert prob.likely_range_lower < prob.likely_range_upper


def test_nps_interval_is_independent_of_scalar_nps_confidence():
    # Identical distribution, wildly different scalar NPS point forecasts.
    low_env = ProductionPredictionAdapter(ScalarNpsService(20.0)).predict(
        {}, simulations=1000, seed=42
    ).nps
    high_env = ProductionPredictionAdapter(ScalarNpsService(95.0)).predict(
        {}, simulations=1000, seed=42
    ).nps
    low = low_env.probabilistic
    high = high_env.probabilistic

    # The point forecast follows the scalar NPS value.
    assert low_env.prediction != high_env.prediction
    # The interval does NOT follow it: it is derived from the 0..10
    # distribution's Monte Carlo, so changing the scalar NPS must not move
    # the interval (i.e. there is no scalar-NPS confidence/uncertainty band).
    assert low.likely_range_lower == high.likely_range_lower
    assert low.likely_range_upper == high.likely_range_upper
