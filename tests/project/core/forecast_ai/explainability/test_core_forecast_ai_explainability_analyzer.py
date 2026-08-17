import importlib

def test_analyzer_surface():
    module = importlib.import_module("core.forecast_ai.explainability.analyzer")
    assert hasattr(module, "analyze_forecast")
    assert hasattr(module, "analyze_trend")
    assert hasattr(module, "analyze_sensitivity")
    assert hasattr(module, "analyze_recommendations")
    assert hasattr(module, "analyze_strategy")
    assert hasattr(module, "analyze_confidence")
    assert hasattr(module, "analyze_risk")
    assert hasattr(module, "ExplainabilityAnalyzer")
