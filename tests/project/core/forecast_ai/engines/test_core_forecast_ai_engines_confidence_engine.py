import importlib

def test_confidence_engine_surface():
    module = importlib.import_module("core.forecast_ai.engines.confidence_engine")
    assert hasattr(module, "execute")
    assert hasattr(module, "ConfidenceEngine")
