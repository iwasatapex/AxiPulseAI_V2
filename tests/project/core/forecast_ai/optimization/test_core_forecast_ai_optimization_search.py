import importlib

def test_search_surface():
    module = importlib.import_module("core.forecast_ai.optimization.search")
    assert hasattr(module, "iterate")
    assert hasattr(module, "DeterministicHillClimb")
