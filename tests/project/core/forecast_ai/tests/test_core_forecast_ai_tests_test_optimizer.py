import importlib

def test_test_optimizer_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_optimizer")
    assert hasattr(module, "predict")
    assert hasattr(module, "setUpClass")
    assert hasattr(module, "setUp")
    assert hasattr(module, "test_oh_only_target")
    assert hasattr(module, "test_nps_only_target")
    assert hasattr(module, "test_constraint_fixed")
    assert hasattr(module, "test_scenario_integration")
    assert hasattr(module, "DummyPredictor")
    assert hasattr(module, "TestOptimizer")
