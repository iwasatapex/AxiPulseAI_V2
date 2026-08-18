import importlib

def test_engine_surface():
    module = importlib.import_module("core.forecast_ai.trends.engine")
    assert hasattr(module, "analyze")
    assert hasattr(module, "TrendEngine")
