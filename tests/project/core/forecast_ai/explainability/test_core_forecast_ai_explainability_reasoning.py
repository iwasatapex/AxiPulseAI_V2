import importlib

def test_reasoning_surface():
    module = importlib.import_module("core.forecast_ai.explainability.reasoning")
    assert hasattr(module, "build_reasoning")
    assert hasattr(module, "ReasoningBuilder")
