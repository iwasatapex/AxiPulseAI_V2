import importlib

import pytest

from api.services.nps_service import NPSService, NPSServiceUnavailableError


def test_nps_service_surface():
    module = importlib.import_module("api.services.nps_service")
    assert hasattr(module, "load_model")
    assert hasattr(module, "is_loaded")
    assert hasattr(module, "predict")
    assert hasattr(module, "NPSService")


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


def test_real_prediction_returns_engine_dict():
    service = NPSService()
    assert service.is_loaded() is True

    result = service.predict(VALID)

    # Real engine schema - not the mock schema.
    assert isinstance(result, dict)
    assert "nps" in result
    assert "score_counts" in result
    assert "bayesian_score_distribution" in result
    assert -100.0 <= result["nps"] <= 100.0
    assert "distribution" not in result
    assert "ensemble_details" not in result


def test_no_mock_fallback_by_default():
    service = NPSService(model_path="models/__does_not_exist__.pkl")
    assert service.is_loaded() is False

    with pytest.raises(NPSServiceUnavailableError):
        service.predict(VALID)


def test_explicit_mock_only_when_requested():
    service = NPSService(model_path="models/__does_not_exist__.pkl")

    # mock=True is the explicit escape hatch; still never the default.
    result = service.predict(VALID, mock=True)
    assert isinstance(result, dict)
    assert "nps" in result


def test_build_prediction_row_defaults_total_surveys():
    row = NPSService._build_prediction_row(dict(VALID))
    assert row["total_surveys"] == 120
    assert "date" in row

    # None-valued optional fields are dropped so engine defaults apply.
    row2 = NPSService._build_prediction_row(
        dict(VALID, total_surveys=None, business_intelligence_factor=None)
    )
    assert row2["total_surveys"] == 120
    assert "business_intelligence_factor" not in row2

