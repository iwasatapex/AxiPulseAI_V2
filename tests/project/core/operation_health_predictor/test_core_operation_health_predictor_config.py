import importlib

def test_config_surface():
    module = importlib.import_module("core.operation_health_predictor.config")
    assert hasattr(module, "Config")
