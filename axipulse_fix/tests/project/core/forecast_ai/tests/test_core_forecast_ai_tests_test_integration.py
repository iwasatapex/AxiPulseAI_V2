import importlib

def test_test_integration_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_integration")
    assert hasattr(module, "test_end_to_end")
    assert hasattr(module, "TestIntegration")
