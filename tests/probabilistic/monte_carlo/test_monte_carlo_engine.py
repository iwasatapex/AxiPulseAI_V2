import importlib

def test_simulator_surface():
    module = importlib.import_module("core.monte_carlo.engine")
    assert hasattr(module, "simulate")
    assert hasattr(module, "MonteCarloResult")
    assert hasattr(module, "MonteCarloEngine")
