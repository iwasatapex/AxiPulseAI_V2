import importlib

def test_engine_surface():
    module = importlib.import_module("core.forecast_ai.sensitivity.engine")
    assert hasattr(module, "analyze")
    assert hasattr(module, "SensitivityEngine")
