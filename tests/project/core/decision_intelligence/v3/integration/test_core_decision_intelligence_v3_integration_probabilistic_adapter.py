import importlib

def test_probabilistic_adapter_surface():
    module = importlib.import_module("core.decision_intelligence.v3.integration.probabilistic_adapter")
    assert hasattr(module, "analyze")
    assert hasattr(module, "analyze_prediction")
    assert hasattr(module, "PredictorProbabilisticResult")
    assert hasattr(module, "UniversalProbabilisticAdapter")
