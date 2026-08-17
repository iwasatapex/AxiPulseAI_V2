import importlib

def test_engine_surface():
    module = importlib.import_module("core.forecast_ai.recommendations.engine")
    assert hasattr(module, "generate")
    assert hasattr(module, "RecommendationEngine")
