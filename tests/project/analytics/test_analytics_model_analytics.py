import importlib

def test_model_analytics_surface():
    module = importlib.import_module("analytics.model_analytics")
    assert hasattr(module, "set_feature_names")
    assert hasattr(module, "evaluate_engine1")
    assert hasattr(module, "evaluate_engine2")
    assert hasattr(module, "evaluate_on_data")
    assert hasattr(module, "feature_importance")
    assert hasattr(module, "cross_validation_scores")
    assert hasattr(module, "ModelAnalytics")
