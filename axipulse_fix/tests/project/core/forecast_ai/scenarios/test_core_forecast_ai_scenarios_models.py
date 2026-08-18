import importlib

def test_models_surface():
    module = importlib.import_module("core.forecast_ai.scenarios.models")
    assert hasattr(module, "is_active")
    assert hasattr(module, "ModifierType")
    assert hasattr(module, "Modifier")
    assert hasattr(module, "Scenario")
