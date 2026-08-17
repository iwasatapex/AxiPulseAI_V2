import importlib

def test_health_metrics_surface():
    module = importlib.import_module("api.monitoring.health_metrics")
    assert hasattr(module, "track_engine")
