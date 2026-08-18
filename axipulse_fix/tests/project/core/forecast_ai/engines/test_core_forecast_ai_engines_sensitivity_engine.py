import importlib

def test_sensitivity_engine_surface():
    module = importlib.import_module("core.forecast_ai.engines.sensitivity_engine")
    assert hasattr(module, "execute")
    assert hasattr(module, "SensitivityEngine")
