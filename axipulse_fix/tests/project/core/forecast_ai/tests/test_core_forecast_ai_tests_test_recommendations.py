import importlib

def test_test_recommendations_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_recommendations")
    assert hasattr(module, "setUp")
    assert hasattr(module, "test_generate_recommendations")
    assert hasattr(module, "test_ranking")
    assert hasattr(module, "test_formatter")
    assert hasattr(module, "test_template_mapping")
    assert hasattr(module, "TestRecommendations")
