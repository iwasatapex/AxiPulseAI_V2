import importlib

def test_prediction_logger_surface():
    module = importlib.import_module("api.services.prediction_logger")
    assert hasattr(module, "log")
    assert hasattr(module, "PredictionLogger")
