import importlib

def test_patterns_surface():
    module = importlib.import_module("core.forecast_ai.trends.patterns")
    assert hasattr(module, "detect")
    assert hasattr(module, "PatternDetector")
