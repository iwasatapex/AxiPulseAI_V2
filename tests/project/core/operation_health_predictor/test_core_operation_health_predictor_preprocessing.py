import importlib

def test_preprocessing_surface():
    module = importlib.import_module("core.operation_health_predictor.preprocessing")
    assert hasattr(module, "load_data")
    assert hasattr(module, "DataLoadingMixin")
