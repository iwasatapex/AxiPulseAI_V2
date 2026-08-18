import importlib

def test_engine_surface():
    module = importlib.import_module("core.forecast_ai.risk.engine")
    assert hasattr(module, "evaluate")
    assert hasattr(module, "RiskEngine")
