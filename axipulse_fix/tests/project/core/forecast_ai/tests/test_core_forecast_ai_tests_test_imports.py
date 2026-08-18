import importlib

def test_test_imports_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_imports")
    assert hasattr(module, "test_import_forecast_ai_does_not_trigger_simulator_or_models")
    assert hasattr(module, "TestImportSideEffects")
