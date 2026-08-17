import importlib


def test_bayesian_engine_surface():
    module = importlib.import_module("core.bayesian.engine")

    assert hasattr(module, "BayesianResult")
    assert hasattr(module, "BayesianInferenceEngine")


def test_bayesian_engine_basic_inference():
    module = importlib.import_module("core.bayesian.engine")

    engine = module.BayesianInferenceEngine()

    result = engine.infer(
        observations=[1.0, 1.0, 0.0, 1.0],
        prior_mean=0.5,
        prior_strength=2.0,
    )

    assert isinstance(result, module.BayesianResult)
    assert 0.0 <= result.probability <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.posterior_mean <= 1.0
    assert result.posterior_std >= 0.0
    assert result.samples == 4


def test_bayesian_engine_empty_observations():
    module = importlib.import_module("core.bayesian.engine")

    engine = module.BayesianInferenceEngine()

    result = engine.infer(
        observations=[],
        prior_mean=0.5,
        prior_strength=2.0,
    )

    assert isinstance(result, module.BayesianResult)
    assert result.probability == 0.5
    assert result.posterior_mean == 0.5
    assert result.samples == 0
