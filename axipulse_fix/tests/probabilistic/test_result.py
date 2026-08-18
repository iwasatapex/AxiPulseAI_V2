"""Focused tests for the universal probabilistic result contract."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from core.probabilistic import (
    BayesianInfo,
    MonteCarloInfo,
    ProbabilisticResult,
    ProbabilisticResultBase,
)


def test_exports_and_alias():
    assert ProbabilisticResult is ProbabilisticResultBase


def test_minimal_result():
    result = ProbabilisticResult()

    assert result.most_likely is None
    assert result.probability_of_target is None
    assert result.bayesian is None
    assert result.monte_carlo is None
    assert result.contract_version == "1.0.0"
    assert isinstance(result.created_at, datetime)


def test_full_result_and_serialization():
    result = ProbabilisticResult(
        most_likely=82.0,
        likely_range_lower=74.0,
        likely_range_upper=89.0,
        range_confidence=0.90,
        probability_of_target=0.78,
        probability_of_failure=0.22,
        expected_value=82.3,
        uncertainty=4.2,
        risk=0.07,
        confidence=0.84,
        bayesian=BayesianInfo(
            posterior_mean=82.0,
            posterior_std=4.2,
            credible_interval_lower=74.0,
            credible_interval_upper=89.0,
            credible_level=0.90,
        ),
        monte_carlo=MonteCarloInfo(
            num_simulations=10000,
            percentile_5=74.0,
            percentile_50=82.0,
            percentile_95=89.0,
            other_percentiles={
                0.10: 76.0,
                0.25: 79.0,
                0.75: 85.0,
                0.90: 88.0,
            },
        ),
    )

    payload = result.model_dump()

    assert payload["most_likely"] == 82.0
    assert payload["likely_range_lower"] == 74.0
    assert payload["likely_range_upper"] == 89.0
    assert payload["probability_of_target"] == 0.78
    assert payload["monte_carlo"]["num_simulations"] == 10000


def test_probability_bounds():
    with pytest.raises(ValidationError):
        ProbabilisticResult(probability_of_target=1.5)

    with pytest.raises(ValidationError):
        ProbabilisticResult(probability_of_failure=-0.1)


def test_range_ordering():
    with pytest.raises(ValidationError):
        ProbabilisticResult(
            likely_range_lower=120.0,
            likely_range_upper=80.0,
        )


def test_bayesian_interval_ordering():
    with pytest.raises(ValidationError):
        BayesianInfo(
            credible_interval_lower=120.0,
            credible_interval_upper=80.0,
        )


def test_monte_carlo_percentile_ordering():
    with pytest.raises(ValidationError):
        MonteCarloInfo(
            percentile_50=110.0,
            percentile_95=100.0,
        )


def test_monte_carlo_percentile_key_bounds():
    with pytest.raises(ValidationError):
        MonteCarloInfo(
            other_percentiles={1.5: 100.0},
        )


def test_optional_range_confidence():
    result = ProbabilisticResult(
        most_likely=50.0,
        likely_range_lower=40.0,
        likely_range_upper=60.0,
    )

    assert result.range_confidence is None
