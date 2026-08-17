import importlib

def test_requests_surface():
    module = importlib.import_module("api.models.requests")
    assert hasattr(module, "validate_release_transfer_sum")
    assert hasattr(module, "HealthPredictRequest")
    assert hasattr(module, "HealthBatchRequest")
    assert hasattr(module, "NPSPredictRequest")
    assert hasattr(module, "NPSBatchRequest")
    assert hasattr(module, "DashboardRequest")
    assert hasattr(module, "SystemStatusRequest")
