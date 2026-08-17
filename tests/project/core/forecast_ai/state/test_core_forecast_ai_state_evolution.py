import importlib

def test_evolution_surface():
    module = importlib.import_module("core.forecast_ai.state.evolution")
    assert hasattr(module, "evolve")
    assert hasattr(module, "StateEvolutionEngine")
