import importlib

def test_config_surface():
    module = importlib.import_module("core.nps_predictor.config")
    assert hasattr(module, "Config")
