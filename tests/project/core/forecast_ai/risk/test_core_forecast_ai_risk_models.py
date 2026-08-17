import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.risk.models")
    assert hasattr(module, "RiskCategory")
    assert hasattr(module, "RiskFactor")
    assert hasattr(module, "RiskAnalysis")
    assert hasattr(module, "RiskResult")
