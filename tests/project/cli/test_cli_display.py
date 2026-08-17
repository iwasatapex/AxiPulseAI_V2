import importlib

def test_display_surface():
    module = importlib.import_module("cli.display")
    assert hasattr(module, "display_ops_metadata")
    assert hasattr(module, "display_nps_metadata")
