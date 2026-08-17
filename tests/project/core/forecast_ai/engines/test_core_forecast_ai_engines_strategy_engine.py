import importlib

def test_strategy_engine_surface():
    module = importlib.import_module("core.forecast_ai.engines.strategy_engine")
    assert hasattr(module, "execute")
    assert hasattr(module, "StrategyEngine")
