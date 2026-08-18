import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.trends.models")
    assert hasattr(module, "TrendSeries")
    assert hasattr(module, "TrendAnalysis")
    assert hasattr(module, "TrendResult")
