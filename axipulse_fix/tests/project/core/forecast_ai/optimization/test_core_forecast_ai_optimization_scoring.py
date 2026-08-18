import importlib

def test_scoring_surface():
    module = importlib.import_module("core.forecast_ai.optimization.scoring")
    assert hasattr(module, "compute_distance")
    assert hasattr(module, "compute_score")
    assert hasattr(module, "is_acceptable")
    assert hasattr(module, "ScoreCalculator")
