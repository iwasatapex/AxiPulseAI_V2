import importlib

def test_test_risk_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_risk")
    assert hasattr(module, "test_risk_score_weighted")
    assert hasattr(module, "test_classification")
    assert hasattr(module, "test_forecast_detector")
    assert hasattr(module, "test_trend_detector")
    assert hasattr(module, "test_sensitivity_detector")
    assert hasattr(module, "test_recommendation_detector")
    assert hasattr(module, "test_strategy_detector")
    assert hasattr(module, "test_confidence_detector")
    assert hasattr(module, "test_formatter")
    assert hasattr(module, "test_component_aggregation_policy_defined")
    assert hasattr(module, "TestRisk")
