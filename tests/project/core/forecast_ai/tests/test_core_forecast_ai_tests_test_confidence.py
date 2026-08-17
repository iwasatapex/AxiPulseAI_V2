import importlib

def test_test_confidence_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_confidence")
    assert hasattr(module, "test_metric_prediction_stability")
    assert hasattr(module, "test_metric_trend_consistency")
    assert hasattr(module, "test_metric_sensitivity_consistency")
    assert hasattr(module, "test_metric_recommendation_agreement")
    assert hasattr(module, "test_metric_strategy_completeness_gradual")
    assert hasattr(module, "test_confidence_scorer")
    assert hasattr(module, "test_classification")
    assert hasattr(module, "test_weighted_overall_confidence")
    assert hasattr(module, "test_formatter")
    assert hasattr(module, "test_missing_components")
    assert hasattr(module, "TestConfidence")
