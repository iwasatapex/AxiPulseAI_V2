from __future__ import annotations

from core.forecast_ai.models import PredictionResult
from core.forecast_ai.prediction.production import (
    ProductionPredictionAdapter,
    ProductionPredictionResult,
)


class StubPredictionService:
    def __init__(self) -> None:
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        return PredictionResult(
            operations_health=82.0,
            nps=81.0,
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
        nps_uncertainty=2.0,
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
