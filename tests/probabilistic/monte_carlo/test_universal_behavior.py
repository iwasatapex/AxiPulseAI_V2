from core.monte_carlo import MonteCarloEngine, MonteCarloResult


def test_monte_carlo_result_structure():
    result = MonteCarloEngine().simulate(
        baseline=80.0,
        uncertainty=2.0,
        samples=2000,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.samples == 2000
    assert result.p05 <= result.p50 <= result.p95


def test_monte_carlo_is_reproducible():
    engine = MonteCarloEngine()

    first = engine.simulate(
        baseline=80.0,
        uncertainty=2.0,
        samples=1000,
    )

    second = engine.simulate(
        baseline=80.0,
        uncertainty=2.0,
        samples=1000,
    )

    assert first == second


def test_monte_carlo_changes_with_uncertainty():
    engine = MonteCarloEngine()

    narrow = engine.simulate(
        baseline=80.0,
        uncertainty=1.0,
        samples=2000,
    )

    wide = engine.simulate(
        baseline=80.0,
        uncertainty=10.0,
        samples=2000,
    )

    narrow_width = narrow.p95 - narrow.p05
    wide_width = wide.p95 - wide.p05

    assert wide_width > narrow_width


def test_monte_carlo_invalid_samples_are_rejected():
    engine = MonteCarloEngine()

    try:
        engine.simulate(
            baseline=80.0,
            samples=0,
        )
    except ValueError:
        return

    raise AssertionError("Invalid sample count was accepted")


def test_monte_carlo_invalid_uncertainty_is_rejected():
    engine = MonteCarloEngine()

    try:
        engine.simulate(
            baseline=80.0,
            uncertainty=-1.0,
        )
    except ValueError:
        return

    raise AssertionError("Negative uncertainty was accepted")
