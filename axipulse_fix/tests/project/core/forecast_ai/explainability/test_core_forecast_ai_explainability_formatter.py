import importlib

def test_formatter_surface():
    module = importlib.import_module("core.forecast_ai.explainability.formatter")
    assert hasattr(module, "to_text")
    assert hasattr(module, "to_markdown")
    assert hasattr(module, "to_dict")
    assert hasattr(module, "to_json")
    assert hasattr(module, "explanation_dict")
    assert hasattr(module, "ExplainabilityFormatter")
