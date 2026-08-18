import importlib

def test_planner_surface():
    module = importlib.import_module("core.forecast_ai.planner")
    assert hasattr(module, "validate")
    assert hasattr(module, "route")
    assert hasattr(module, "dispatch")
    assert hasattr(module, "build")
    assert hasattr(module, "execute")
    assert hasattr(module, "validate")
    assert hasattr(module, "route")
    assert hasattr(module, "Validator")
    assert hasattr(module, "Router")
    assert hasattr(module, "Dispatcher")
    assert hasattr(module, "ResponseBuilder")
    assert hasattr(module, "ForecastAIPlanner")
