import importlib

def test_statistics_surface():
    module = importlib.import_module("core.forecast_ai.trends.statistics")
    assert hasattr(module, "mean")
    assert hasattr(module, "median")
    assert hasattr(module, "variance")
    assert hasattr(module, "std_dev")
    assert hasattr(module, "slope")
    assert hasattr(module, "moving_average")
    assert hasattr(module, "absolute_change")
    assert hasattr(module, "percent_change")
    assert hasattr(module, "Statistics")
