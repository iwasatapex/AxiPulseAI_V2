import importlib

def test_predictor_surface():
    module = importlib.import_module("core.operation_health_predictor.predictor")
    assert hasattr(module, "OperationalHealthPredictor")
