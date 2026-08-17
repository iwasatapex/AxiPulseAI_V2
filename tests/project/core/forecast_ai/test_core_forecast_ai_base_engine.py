import importlib

def test_base_engine_surface():
    module = importlib.import_module("core.forecast_ai.base_engine")
    assert hasattr(module, "execute")
    assert hasattr(module, "ForecastAIEngine")
