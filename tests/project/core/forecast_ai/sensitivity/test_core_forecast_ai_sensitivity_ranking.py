import importlib

def test_ranking_surface():
    module = importlib.import_module("core.forecast_ai.sensitivity.ranking")
    assert hasattr(module, "rank")
    assert hasattr(module, "SensitivityRanker")
