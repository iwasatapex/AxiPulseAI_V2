import importlib

def test_metrics_surface():
    module = importlib.import_module("api.middleware.metrics")
    assert hasattr(module, "dispatch")
    assert hasattr(module, "MetricsMiddleware")
