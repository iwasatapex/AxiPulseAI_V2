import importlib

def test_utils_surface():
    module = importlib.import_module("cli.utils")
    assert hasattr(module, "bounded")
