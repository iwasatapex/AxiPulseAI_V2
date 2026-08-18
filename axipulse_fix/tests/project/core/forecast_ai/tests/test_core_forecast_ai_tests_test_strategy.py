import importlib

def test_test_strategy_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_strategy")
    assert hasattr(module, "setUp")
    assert hasattr(module, "test_strategy_generation")
    assert hasattr(module, "test_deterministic_ids")
    assert hasattr(module, "test_grouping")
    assert hasattr(module, "test_timeline_generation")
    assert hasattr(module, "test_scoring")
    assert hasattr(module, "test_formatter")
    assert hasattr(module, "TestStrategy")
