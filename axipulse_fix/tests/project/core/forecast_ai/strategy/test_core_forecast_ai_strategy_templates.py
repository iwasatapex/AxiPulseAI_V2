import importlib

def test_templates_surface():
    module = importlib.import_module("core.forecast_ai.strategy.templates")
    assert hasattr(module, "get_template")
    assert hasattr(module, "get_priority")
    assert hasattr(module, "StrategyTemplates")
