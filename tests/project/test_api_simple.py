import importlib

def test_api_simple_surface():
    module = importlib.import_module("api_simple")
    assert hasattr(module, "root")
    assert hasattr(module, "health")
    assert hasattr(module, "adie_decision")
