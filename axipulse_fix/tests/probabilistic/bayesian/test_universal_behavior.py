from core.bayesian import BayesianInferenceEngine, BayesianResult


def test_bayesian_result_is_bounded():
    result = BayesianInferenceEngine().infer(
        observations=[1.0, 0.0, 1.0, 1.0]
    )

    assert isinstance(result, BayesianResult)
    assert 0.0 <= result.probability <= 1.0
    assert 0.0 <= result.posterior_mean <= 1.0
    assert result.posterior_std >= 0.0
    assert 0.0 <= result.confidence <= 1.0


def test_bayesian_prior_affects_result():
    engine = BayesianInferenceEngine()

    low_prior = engine.infer(
        observations=[],
        prior_mean=0.2,
        prior_strength=10.0,
    )

    high_prior = engine.infer(
        observations=[],
        prior_mean=0.8,
        prior_strength=10.0,
    )

    assert low_prior.posterior_mean < high_prior.posterior_mean


def test_bayesian_more_evidence_updates_belief():
    engine = BayesianInferenceEngine()

    weak = engine.infer(
        observations=[0.0, 0.0]
    )

    strong = engine.infer(
        observations=[1.0, 1.0, 1.0, 1.0]
    )

    assert strong.posterior_mean > weak.posterior_mean


def test_bayesian_invalid_prior_is_rejected():
    engine = BayesianInferenceEngine()

    try:
        engine.infer(
            observations=[1.0],
            prior_mean=1.5,
        )
    except (ValueError, AssertionError):
        return

    raise AssertionError("Invalid prior_mean was accepted")
