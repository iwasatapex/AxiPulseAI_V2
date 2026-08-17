import importlib

def test_manager_surface():
    module = importlib.import_module("core.forecast_ai.scenarios.manager")
    assert hasattr(module, "get_active_scenarios")
    assert hasattr(module, "apply_scenarios")
    assert hasattr(module, "apply_scenarios_to_state")
    assert hasattr(module, "validate_scenario")
    assert hasattr(module, "ScenarioManager")
