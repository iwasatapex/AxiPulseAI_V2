import importlib

def test_planner_surface():
    module = importlib.import_module("core.forecast_ai.strategy.planner")
    assert hasattr(module, "group_recommendations")
    assert hasattr(module, "StrategyPlanner")
