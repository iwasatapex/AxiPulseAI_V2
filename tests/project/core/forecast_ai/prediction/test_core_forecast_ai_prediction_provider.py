import importlib

def test_provider_surface():
    module = importlib.import_module("core.forecast_ai.prediction.provider")
    assert hasattr(module, "get_oh_predictor")
    assert hasattr(module, "get_nps_predictor")
    assert hasattr(module, "set_oh_predictor")
    assert hasattr(module, "set_nps_predictor")
    assert hasattr(module, "reset")
    assert hasattr(module, "PredictorProvider")
