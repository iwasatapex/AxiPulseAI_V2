from core.decision_intelligence.v3.intelligence import (
    ADIEProbabilisticEngine,
)


def test_bayesian_inference():
    engine = ADIEProbabilisticEngine()

    result = engine.analyze(
        observations=[1, 1, 1, 0, 1],
        baseline=0.8,
        samples=2000,
    )

    assert 0.0 <= result.bayesian.probability <= 1.0
    assert 0.0 <= result.bayesian.confidence <= 1.0
    assert result.bayesian.samples == 5


def test_monte_carlo():
    engine = ADIEProbabilisticEngine()

    result = engine.analyze(
        observations=[1, 1, 0, 1],
        baseline=0.8,
        uncertainty=0.05,
        samples=2000,
    )

    assert result.monte_carlo.samples == 2000
    assert result.monte_carlo.p05 < result.monte_carlo.p95


def test_deterministic_seed():
    engine = ADIEProbabilisticEngine()

    a = engine.analyze(
        observations=[1, 0, 1],
        baseline=0.7,
        samples=1000,
    )

    b = engine.analyze(
        observations=[1, 0, 1],
        baseline=0.7,
        samples=1000,
    )

    assert a.monte_carlo.mean == b.monte_carlo.mean


# ---------------------------------------------------------------------------
# Probability-domain bound invariant (decision-level Monte Carlo).
# ---------------------------------------------------------------------------

def _assert_probability_domain(mc):
    assert 0.0 <= mc.p05 <= 1.0
    assert 0.0 <= mc.p50 <= 1.0
    assert 0.0 <= mc.p95 <= 1.0
    assert 0.0 <= mc.mean <= 1.0
    assert mc.p05 <= mc.p50 <= mc.p95
    # Same-draw invariant: counts partition the same sample; bins total to it.
    assert mc.success_count + mc.failure_count == mc.samples
    assert sum(b["count"] for b in mc.distribution) == mc.samples


def test_probability_domain_never_exceeds_one():
    """baseline=0.95, uncertainty=0.05 must not produce upside >1.0."""
    engine = ADIEProbabilisticEngine()
    result = engine.analyze(
        observations=[1, 1, 1, 0, 1],
        baseline=0.95,
        uncertainty=0.05,
        samples=20000,
    )
    _assert_probability_domain(result.monte_carlo)


def test_probability_domain_at_boundaries():
    """baseline=0 and baseline=1 stay inside [0,1]."""
    engine = ADIEProbabilisticEngine()
    low = engine.analyze(
        observations=[0, 0, 0, 0], baseline=0.0, uncertainty=0.05, samples=5000
    ).monte_carlo
    high = engine.analyze(
        observations=[1, 1, 1, 1], baseline=1.0, uncertainty=0.05, samples=5000
    ).monte_carlo
    _assert_probability_domain(low)
    _assert_probability_domain(high)


def test_probability_domain_low_and_high_uncertainty():
    engine = ADIEProbabilisticEngine()
    for uncertainty in (0.001, 0.5, 1.0):
        mc = engine.analyze(
            observations=[1, 1, 0, 1],
            baseline=0.7,
            uncertainty=uncertainty,
            samples=5000,
        ).monte_carlo
        _assert_probability_domain(mc)


def test_probability_domain_deterministic_repeat():
    engine = ADIEProbabilisticEngine()
    a = engine.analyze(
        observations=[1, 1, 0, 1],
        baseline=0.95,
        uncertainty=0.05,
        samples=2000,
    ).monte_carlo
    b = engine.analyze(
        observations=[1, 1, 0, 1],
        baseline=0.95,
        uncertainty=0.05,
        samples=2000,
    ).monte_carlo
    assert (a.p05, a.p50, a.p95, a.mean, a.success_count) == (
        b.p05, b.p50, b.p95, b.mean, b.success_count,
    )
