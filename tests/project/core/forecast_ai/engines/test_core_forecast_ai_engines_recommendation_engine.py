import importlib

def test_recommendation_engine_surface():
    module = importlib.import_module("core.forecast_ai.engines.recommendation_engine")
    assert hasattr(module, "execute")
    assert hasattr(module, "RecommendationEngine")
