import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.sensitivity.models")
    assert hasattr(module, "SensitivityExperiment")
    assert hasattr(module, "SensitivityAnalysis")
    assert hasattr(module, "SensitivityResult")
