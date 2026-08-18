import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.strategy.models")
    assert hasattr(module, "StrategyCategory")
    assert hasattr(module, "Milestone")
    assert hasattr(module, "StrategyPlan")
    assert hasattr(module, "StrategyResult")
