import importlib

def test_engine_surface():
    module = importlib.import_module("core.forecast_ai.explainability.engine")
    assert hasattr(module, "explain")
    assert hasattr(module, "ExplainabilityEngine")
