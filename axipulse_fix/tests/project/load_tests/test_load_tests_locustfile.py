import importlib

def test_locustfile_surface():
    module = importlib.import_module("load_tests.locustfile")
    assert hasattr(module, "health")
    assert hasattr(module, "system_status")
    assert hasattr(module, "metrics")
    assert hasattr(module, "adie")
    assert hasattr(module, "AxiPulseUser")
