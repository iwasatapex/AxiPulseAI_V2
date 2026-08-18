import importlib

def test_test_contract_surface():
    module = importlib.import_module("core.forecast_ai.tests.test_contract")
    assert hasattr(module, "test_all_engines_have_execute_and_return_response")
    assert hasattr(module, "TestEngineContract")
