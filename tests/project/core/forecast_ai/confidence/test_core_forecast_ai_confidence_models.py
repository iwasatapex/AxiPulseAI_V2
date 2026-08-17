import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.confidence.models")
    assert hasattr(module, "ConfidenceMetric")
    assert hasattr(module, "ConfidenceAnalysis")
    assert hasattr(module, "ConfidenceResult")
