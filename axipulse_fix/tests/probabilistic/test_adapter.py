from core.probabilistic import (
    ProbabilisticResult,
    UniversalProbabilisticAdapter,
)


def test_bayesian_adapter_returns_universal_result():
    adapter = UniversalProbabilisticAdapter()

    result = adapter.from_bayesian(
        observations=[1.0, 1.0, 0.0, 1.0]
    )

    assert isinstance(result, ProbabilisticResult)
    assert result.bayesian is not None
    assert result.monte_carlo is None
    assert result.likely_range_lower <= result.most_likely
    assert result.most_likely <= result.likely_range_upper


def test_monte_carlo_adapter_returns_universal_result():
    adapter = UniversalProbabilisticAdapter()

    result = adapter.from_monte_carlo(
        baseline=80.0,
        uncertainty=2.0,
        samples=1000,
        seed=42,
    )

    assert isinstance(result, ProbabilisticResult)
    assert result.monte_carlo is not None
    assert result.bayesian is None
    assert result.likely_range_lower <= result.most_likely
    assert result.most_likely <= result.likely_range_upper


def test_monte_carlo_adapter_is_reproducible():
    adapter = UniversalProbabilisticAdapter()

    first = adapter.from_monte_carlo(
        baseline=80.0,
        uncertainty=2.0,
        samples=1000,
        seed=42,
    )

    second = adapter.from_monte_carlo(
        baseline=80.0,
        uncertainty=2.0,
        samples=1000,
        seed=42,
    )

    first_data = first.model_dump()
    second_data = second.model_dump()
    first_data.pop("created_at", None)
    second_data.pop("created_at", None)
    assert first_data == second_data
