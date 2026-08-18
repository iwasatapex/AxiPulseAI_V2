import importlib

def test_metrics_surface():
    module = importlib.import_module("core.forecast_ai.validation.metrics")
    assert hasattr(module, "mae")
    assert hasattr(module, "bias")
    assert hasattr(module, "rmse")
    assert hasattr(module, "ForecastMetrics")
