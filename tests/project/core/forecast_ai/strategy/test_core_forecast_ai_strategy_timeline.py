import importlib

def test_timeline_surface():
    module = importlib.import_module("core.forecast_ai.strategy.timeline")
    assert hasattr(module, "generate")
    assert hasattr(module, "TimelineGenerator")
