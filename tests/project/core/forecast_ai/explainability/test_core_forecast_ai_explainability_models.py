import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.explainability.models")
    assert hasattr(module, "Evidence")
    assert hasattr(module, "ExplanationTrace")
    assert hasattr(module, "Explanation")
    assert hasattr(module, "ExplainabilityResult")
