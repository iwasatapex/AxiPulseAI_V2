"""Tests for the canonical V3 ADIE service surface and behavior."""

import importlib

import pytest

from api.services.adie_v3_service import ADIEV3Service


def test_adie_v3_service_surface():
    module = importlib.import_module("api.services.adie_v3_service")
    assert hasattr(module, "ADIEV3Service")
    assert hasattr(module, "analyze")
    assert hasattr(module, "analyze_scenarios")


def test_analyze_valid_inputs():
    service = ADIEV3Service()
    result = service.analyze(
        observations=[1, 1, 0, 1, 1],
        baseline=0.8,
        samples=2000,
    )
    assert "probabilistic" in result
    assert "risk" in result
    assert "decision" in result
    assert "explanation" in result


def test_analyze_rejects_future_data():
    service = ADIEV3Service()
    with pytest.raises(ValueError):
        service.analyze(
            observations=[1, 1, 0],
            baseline=0.8,
            samples=1000,
            cutoff="2026-08-10T00:00:00+00:00",
            metadata={"provenance": "2026-08-11T00:00:00+00:00"},
        )


def test_analyze_scenarios_enforces_boundary():
    service = ADIEV3Service()
    with pytest.raises(ValueError):
        service.analyze_scenarios(
            scenarios=[],
            observations=[1, 0, 1],
            baseline=0.8,
            samples=1000,
        )
