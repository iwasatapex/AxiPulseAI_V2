import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.recommendations.models")
    assert hasattr(module, "Category")
    assert hasattr(module, "Priority")
    assert hasattr(module, "Difficulty")
    assert hasattr(module, "Recommendation")
    assert hasattr(module, "RecommendationResult")
