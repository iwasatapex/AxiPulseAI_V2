import importlib

def test_test_sensitivity_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_sensitivity")
    assert hasattr(module, "predict")
    assert hasattr(module, "setUpClass")
    assert hasattr(module, "setUp")
    assert hasattr(module, "test_symmetric_experiments")
    assert hasattr(module, "test_bounds_respected")
    assert hasattr(module, "test_analyzer_aggregation")
    assert hasattr(module, "test_engine_full")
    assert hasattr(module, "test_formatter")
    assert hasattr(module, "DummyPredictor")
    assert hasattr(module, "TestSensitivity")
