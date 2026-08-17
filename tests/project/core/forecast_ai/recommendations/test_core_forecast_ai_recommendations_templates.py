import importlib

def test_templates_surface():
    module = importlib.import_module("core.forecast_ai.recommendations.templates")
    assert hasattr(module, "get_template")
    assert hasattr(module, "get_actions")
    assert hasattr(module, "get_difficulty")
    assert hasattr(module, "RecommendationTemplates")
