import importlib

def test_metrics_routes_surface():
    module = importlib.import_module("api.routes.metrics_routes")
    assert hasattr(module, "metrics")
