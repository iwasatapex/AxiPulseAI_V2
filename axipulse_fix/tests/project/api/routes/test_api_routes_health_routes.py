import importlib

def test_health_routes_surface():
    module = importlib.import_module("api.routes.health_routes")
    assert hasattr(module, "predict_health")
    assert hasattr(module, "health_status")
