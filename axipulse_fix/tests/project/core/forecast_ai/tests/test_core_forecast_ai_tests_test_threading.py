import importlib

def test_test_threading_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_threading")
    assert hasattr(module, "test_multiple_planners_independent")
    assert hasattr(module, "run")
    assert hasattr(module, "TestThreadSafety")
