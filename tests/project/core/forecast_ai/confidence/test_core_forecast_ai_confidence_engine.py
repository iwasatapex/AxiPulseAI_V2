import importlib

def test_engine_surface():
    module = importlib.import_module("core.forecast_ai.confidence.engine")
    assert hasattr(module, "evaluate")
    assert hasattr(module, "ConfidenceEngine")
