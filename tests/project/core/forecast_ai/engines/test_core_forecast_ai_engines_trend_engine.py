import importlib

def test_trend_engine_surface():
    module = importlib.import_module("core.forecast_ai.engines.trend_engine")
    assert hasattr(module, "execute")
    assert hasattr(module, "TrendEngine")
