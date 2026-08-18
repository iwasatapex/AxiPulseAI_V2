import importlib

def test_metrics_surface():
    module = importlib.import_module("core.nps_predictor.metrics")
    assert hasattr(module, "compute_nps_error")
    assert hasattr(module, "calculate_validation_score")
