import importlib

def test_metrics_surface():
    module = importlib.import_module("api.monitoring.metrics")
    assert hasattr(module, "record_engine_call")
