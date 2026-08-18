import importlib

def test_defaults_surface():
    module = importlib.import_module("cli.defaults")
    assert hasattr(module, "load_defaults")
    assert hasattr(module, "save_defaults")
    assert hasattr(module, "display_defaults")
    assert hasattr(module, "update_defaults_from_args")
