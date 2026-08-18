import importlib

def test_test_recursive_forecast_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_recursive_forecast")
    assert hasattr(module, "setUp")
    assert hasattr(module, "test_horizon_1")
    assert hasattr(module, "test_horizon_7")
    assert hasattr(module, "test_horizon_30")
    assert hasattr(module, "test_horizon_365")
    assert hasattr(module, "test_dates_increase")
    assert hasattr(module, "TestRecursiveForecast")
