import importlib

def test_backtester_surface():
    module = importlib.import_module("core.forecast_ai.validation.backtester")
    assert hasattr(module, "evaluate")
    assert hasattr(module, "ForecastBacktester")
