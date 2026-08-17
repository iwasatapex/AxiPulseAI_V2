import importlib

def test_conflicts_surface():
    module = importlib.import_module("core.forecast_ai.recommendations.conflicts")
    assert hasattr(module, "detect_conflicts")
    assert hasattr(module, "ConflictDetector")
