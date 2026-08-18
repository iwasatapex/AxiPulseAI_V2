import importlib

def test_scoring_surface():
    module = importlib.import_module("core.forecast_ai.strategy.scoring")
    assert hasattr(module, "score")
    assert hasattr(module, "rank")
    assert hasattr(module, "StrategyScorer")
