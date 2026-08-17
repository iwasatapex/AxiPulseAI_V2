"""Tests for ADIE V3 API route availability and behavior."""

import importlib


def test_v3_adie_route_imports():
    module = importlib.import_module("api.routes.adie_v3_routes")
    assert hasattr(module, "router")


def test_v3_request_model_finite_validation():
    from api.models.adie_v3_requests import ADIEV3DecisionRequest

    import pytest

    # Valid payload
    req = ADIEV3DecisionRequest(
        observations=[0.1, 0.2, 0.3],
        baseline=0.8,
    )
    assert req.baseline == 0.8
    assert len(req.observations) == 3

    # Non-finite observation must be rejected
    with pytest.raises(Exception):
        ADIEV3DecisionRequest(
            observations=[float("nan"), 0.2],
            baseline=0.8,
        )
