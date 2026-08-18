import importlib

def test_risk_engine_surface():
    module = importlib.import_module("core.forecast_ai.engines.risk_engine")
    assert hasattr(module, "execute")
    assert hasattr(module, "RiskEngine")
