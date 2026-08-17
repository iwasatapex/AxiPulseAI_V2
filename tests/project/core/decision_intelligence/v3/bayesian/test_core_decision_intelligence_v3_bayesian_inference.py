import importlib

def test_inference_surface():
    module = importlib.import_module("core.decision_intelligence.v3.bayesian.inference")
    assert hasattr(module, "infer")
    assert hasattr(module, "BayesianResult")
    assert hasattr(module, "BayesianInferenceEngine")
