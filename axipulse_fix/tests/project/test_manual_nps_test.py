import importlib

def test_manual_nps_test_surface():
    module = importlib.import_module("manual_nps_test")
    assert hasattr(module, "ask")
    assert hasattr(module, "section")
    assert hasattr(module, "main")
