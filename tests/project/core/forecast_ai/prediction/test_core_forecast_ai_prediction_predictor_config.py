import importlib

def test_predictor_config_surface():
    module = importlib.import_module("core.forecast_ai.prediction.predictor_config")
    assert hasattr(module, "create_oh_predictor")
    assert hasattr(module, "create_nps_predictor")
