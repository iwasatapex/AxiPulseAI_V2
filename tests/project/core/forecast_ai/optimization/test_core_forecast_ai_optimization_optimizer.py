import importlib

def test_optimizer_surface():
    module = importlib.import_module("core.forecast_ai.optimization.optimizer")
    assert hasattr(module, "optimize")
    assert hasattr(module, "ReverseOptimizer")
