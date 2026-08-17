import importlib

def test_validation_surface():
    module = importlib.import_module("core.operation_health_predictor.validation")
    assert hasattr(module, "validate_training")
    assert hasattr(module, "validate_prediction")
    assert hasattr(module, "ValidationMixin")
