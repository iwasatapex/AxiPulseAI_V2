import importlib

def test_utils_surface():
    module = importlib.import_module("core.operation_health_predictor.utils")
    assert hasattr(module, "tqdm_joblib")
