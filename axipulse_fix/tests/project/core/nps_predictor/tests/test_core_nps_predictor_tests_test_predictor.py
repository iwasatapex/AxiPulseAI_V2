import importlib

def test_test_predictor_surface():
    module = importlib.import_module("core.nps_predictor.tests.test_predictor")
    assert hasattr(module, "test_import")
    assert hasattr(module, "TestNPSPredictor")
