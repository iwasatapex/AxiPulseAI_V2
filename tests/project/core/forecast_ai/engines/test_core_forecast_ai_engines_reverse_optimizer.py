import importlib

def test_reverse_optimizer_surface():
    module = importlib.import_module("core.forecast_ai.engines.reverse_optimizer")
    assert hasattr(module, "execute")
    assert hasattr(module, "ReverseOptimizer")
