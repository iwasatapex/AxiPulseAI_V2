import importlib

def test_cli_surface():
    module = importlib.import_module("analytics.cli")
    assert hasattr(module, "main")
