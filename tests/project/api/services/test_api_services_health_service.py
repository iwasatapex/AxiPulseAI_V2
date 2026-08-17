import importlib

def test_health_service_surface():
    module = importlib.import_module("api.services.health_service")
    assert hasattr(module, "load_model")
    assert hasattr(module, "is_loaded")
    assert hasattr(module, "predict")
    assert hasattr(module, "HealthService")
