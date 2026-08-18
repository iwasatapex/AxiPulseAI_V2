import importlib

def test_engine_surface():
    module = importlib.import_module("core.decision_intelligence.v3.scenario.engine")
    assert hasattr(module, "to_dict")
    assert hasattr(module, "run")
    assert hasattr(module, "compare")
    assert hasattr(module, "Scenario")
    assert hasattr(module, "ScenarioResult")
    assert hasattr(module, "ADIEScenarioEngine")
