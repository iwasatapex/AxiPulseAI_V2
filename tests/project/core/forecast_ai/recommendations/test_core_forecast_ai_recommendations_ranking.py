import importlib

def test_ranking_surface():
    module = importlib.import_module("core.forecast_ai.recommendations.ranking")
    assert hasattr(module, "rank")
    assert hasattr(module, "key_func")
    assert hasattr(module, "RecommendationRanker")
