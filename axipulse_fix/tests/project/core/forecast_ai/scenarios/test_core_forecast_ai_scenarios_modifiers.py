import importlib

def test_modifiers_surface():
    module = importlib.import_module("core.forecast_ai.scenarios.modifiers")
    assert hasattr(module, "apply_modifier")
    assert hasattr(module, "apply_modifiers")
    assert hasattr(module, "merge_modifiers")
    assert hasattr(module, "validate_modifier")
