import asyncio
import importlib

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import api.routes.nps_routes as nps_routes
from api.models.requests import NPSPredictRequest
from api.models.responses import NPSPredictResponse
from api.services.nps_service import NPSService, NPSServiceUnavailableError


def test_nps_routes_surface():
    module = importlib.import_module("api.routes.nps_routes")
    assert hasattr(module, "predict_nps")
    assert hasattr(module, "nps_status")


VALID = {
    "operational_health": 95.0,
    "target_quality": 87.0,
    "actual_quality": 87.0,
    "target_competency": 93.0,
    "actual_competency": 93.0,
    "target_attendance": 90.0,
    "actual_attendance": 90.0,
    "target_release_rate": 60.0,
    "actual_release_rate": 60.0,
    "target_transfer_rate": 9.0,
    "actual_transfer_rate": 9.0,
    "total_calls_received": 2000,
}


def test_valid_request_returns_real_prediction(monkeypatch):
    service = NPSService()
    assert service.is_loaded()
    monkeypatch.setattr(nps_routes, "_service", service)

    resp = asyncio.run(nps_routes.predict_nps(NPSPredictRequest(**VALID)))

    assert resp["status"] == "success"
    data = resp["data"]
    assert -100.0 <= data["nps"] <= 100.0

    # Real engine fields, not fabricated mock fields.
    for key in (
        "nps",
        "promoters",
        "passives",
        "detractors",
        "score_counts",
        "bayesian_score_distribution",
        "confidence",
        "prediction_interval",
    ):
        assert key in data
    assert "distribution" not in data
    assert "ensemble_details" not in data

    # Cross-check: equals the canonical engine's own output for the same row.
    direct = service.predictor.predict(
        NPSService._build_prediction_row(dict(VALID))
    )
    assert data["nps"] == pytest.approx(direct["nps"])

    # Response validates against the response schema.
    validated = NPSPredictResponse(**resp)
    assert validated.status == "success"


def test_missing_fields_rejected():
    with pytest.raises(ValidationError):
        NPSPredictRequest.model_validate({})


def test_out_of_range_values_rejected():
    with pytest.raises(ValidationError):
        NPSPredictRequest.model_validate({**VALID, "operational_health": 500.0})
    with pytest.raises(ValidationError):
        NPSPredictRequest.model_validate({**VALID, "actual_quality": -1.0})
    with pytest.raises(ValidationError):
        NPSPredictRequest.model_validate({**VALID, "total_calls_received": 0})


def test_model_load_failure_returns_503(monkeypatch):
    service = NPSService(model_path="models/__missing_nps_model__.pkl")
    assert not service.is_loaded()
    monkeypatch.setattr(nps_routes, "_service", service)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(nps_routes.predict_nps(NPSPredictRequest(**VALID)))
    assert exc.value.status_code == 503


def test_no_mock_fallback_in_route():
    service = NPSService(model_path="models/__missing_nps_model__.pkl")
    with pytest.raises(NPSServiceUnavailableError):
        service.predict(VALID)


def test_nps_range_stays_within_100(monkeypatch):
    service = NPSService()
    monkeypatch.setattr(nps_routes, "_service", service)

    for overrides in (
        {"operational_health": 0.0, "actual_quality": 0.0},
        {"operational_health": 120.0, "actual_quality": 100.0},
    ):
        resp = asyncio.run(
            nps_routes.predict_nps(NPSPredictRequest(**{**VALID, **overrides}))
        )
        assert -100.0 <= resp["data"]["nps"] <= 100.0

