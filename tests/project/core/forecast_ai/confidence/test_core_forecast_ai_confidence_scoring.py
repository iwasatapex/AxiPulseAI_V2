import importlib

def test_scoring_surface():
    module = importlib.import_module("core.forecast_ai.confidence.scoring")
    assert hasattr(module, "compute_confidence")
    assert hasattr(module, "classify")
    assert hasattr(module, "ConfidenceScorer")
