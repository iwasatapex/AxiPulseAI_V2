import importlib

def test_forecast_cli_surface():
    module = importlib.import_module("forecast_cli")
    assert hasattr(module, "banner")
    assert hasattr(module, "get_state")
    assert hasattr(module, "run_forecast")
    assert hasattr(module, "run_scenario")
    assert hasattr(module, "run_risk")
    assert hasattr(module, "main")
