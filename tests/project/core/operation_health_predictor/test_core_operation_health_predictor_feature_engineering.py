import importlib

def test_feature_engineering_surface():
    module = importlib.import_module("core.operation_health_predictor.feature_engineering")
    assert hasattr(module, "prepare_features")
    assert hasattr(module, "FeatureEngineeringMixin")
