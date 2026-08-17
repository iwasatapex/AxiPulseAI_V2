import importlib

def test_analyzer_surface():
    module = importlib.import_module("core.forecast_ai.trends.analyzer")
    assert hasattr(module, "analyze")
    assert hasattr(module, "TrendAnalyzer")
