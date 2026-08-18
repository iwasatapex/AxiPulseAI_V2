import importlib

def test_validation_surface():
    module = importlib.import_module("core.nps_predictor.validation")
    assert hasattr(module, "detect_drift")
    assert hasattr(module, "needs_retraining")
