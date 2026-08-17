import importlib

def test_explainability_engine_surface():
    module = importlib.import_module("core.forecast_ai.engines.explainability_engine")
    assert hasattr(module, "execute")
    assert hasattr(module, "ExplainabilityEngine")
