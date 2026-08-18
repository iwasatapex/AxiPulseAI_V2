import importlib

def test_experiments_surface():
    module = importlib.import_module("core.forecast_ai.sensitivity.experiments")
    assert hasattr(module, "generate_experiments")
    assert hasattr(module, "ExperimentGenerator")
