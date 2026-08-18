import importlib

def test_responses_surface():
    module = importlib.import_module("api.models.responses")
    assert hasattr(module, "HealthResponse")
    assert hasattr(module, "NPSResponse")
    assert hasattr(module, "DashboardResponse")
    assert hasattr(module, "HealthPredictResponse")
    assert hasattr(module, "HealthBatchResponse")
    assert hasattr(module, "NPSPredictResponse")
    assert hasattr(module, "NPSBatchResponse")
