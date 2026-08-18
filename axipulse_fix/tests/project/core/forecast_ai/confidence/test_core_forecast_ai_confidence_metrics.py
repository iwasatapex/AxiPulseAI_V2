import importlib

def test_metrics_surface():
    module = importlib.import_module("core.forecast_ai.confidence.metrics")
    assert hasattr(module, "prediction_stability")
    assert hasattr(module, "trend_consistency")
    assert hasattr(module, "sensitivity_consistency")
    assert hasattr(module, "recommendation_agreement")
    assert hasattr(module, "strategy_completeness")
    assert hasattr(module, "cv")
    assert hasattr(module, "ConfidenceMetrics")
