"""Regression tests for the enforced V3 production decision boundary."""

import pytest

from core.decision_intelligence.v3.integration.production_boundary import (
    ProductionDecisionBoundary,
)


BOUNDARY = ProductionDecisionBoundary()


def test_accepts_valid_inputs():
    boundary = ProductionDecisionBoundary()

    result = boundary.decision_service.analyze(
        scenarios=[
            {
                "name": "current_state",
                "probability": 0.7,
                "confidence": 0.6,
                "expected": 0.8,
                "p05": 0.75,
                "p95": 0.85,
            }
        ],
        observations=[1, 1, 0],
        baseline=0.8,
        uncertainty=0.05,
        samples=1000,
    )

    assert result.recommendation == "current_state"
    assert result.risk in {"LOW", "MEDIUM", "HIGH"}


def test_rejects_empty_observations():
    with pytest.raises(ValueError):
        BOUNDARY.validate(observations=[], baseline=0.8)


def test_rejects_non_finite_observations():
    with pytest.raises(ValueError):
        BOUNDARY.validate(observations=[1, float("nan"), 0], baseline=0.8)


def test_rejects_non_finite_baseline():
    with pytest.raises(ValueError):
        BOUNDARY.validate(observations=[1, 0, 1], baseline=float("inf"))


def test_rejects_empty_scenarios():
    with pytest.raises(ValueError):
        BOUNDARY.validate(
            observations=[1, 0, 1], baseline=0.8, scenarios=[]
        )


def test_rejects_predicted_state_treated_as_observed():
    with pytest.raises(ValueError):
        BOUNDARY.validate(
            observations=[1, 0, 1],
            baseline=0.8,
            metadata={
                "is_predicted": True,
                "treated_as_observed": True,
            },
        )


def test_rejects_future_observed_input_with_cutoff():
    # A future-dated observed outcome must be rejected at the boundary.
    with pytest.raises(ValueError):
        BOUNDARY.validate(
            observations=[1, 0, 1],
            baseline=0.8,
            cutoff="2026-08-10T00:00:00+00:00",
            metadata={
                "provenance": "2026-08-11T00:00:00+00:00",
            },
        )


def test_rejects_missing_provenance_when_cutoff_supplied():
    # Safe default: refuse un-verified inputs when a cutoff is given.
    with pytest.raises(ValueError):
        BOUNDARY.validate(
            observations=[1, 0, 1],
            baseline=0.8,
            cutoff="2026-08-10T00:00:00+00:00",
            metadata={},
        )


def test_accepts_past_cutoff_known_input():
    # Inputs known at or before the cutoff are allowed.
    BOUNDARY.validate(
        observations=[1, 0, 1],
        baseline=0.8,
        cutoff="2026-08-11T00:00:00+00:00",
        metadata={
            "provenance": "2026-08-10T00:00:00+00:00",
        },
    )
