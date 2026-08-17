import importlib

def test_registry_surface():
    module = importlib.import_module("core.forecast_ai.scenarios.registry")
    assert hasattr(module, "register")
    assert hasattr(module, "get")
    assert hasattr(module, "list")
    assert hasattr(module, "get_active")
    assert hasattr(module, "reset")
    assert hasattr(module, "ScenarioRegistry")
