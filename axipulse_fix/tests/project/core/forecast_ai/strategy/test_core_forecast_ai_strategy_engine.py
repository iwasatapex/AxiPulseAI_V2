import importlib

def test_engine_surface():
    module = importlib.import_module("core.forecast_ai.strategy.engine")
    assert hasattr(module, "generate")
    assert hasattr(module, "StrategyEngine")
