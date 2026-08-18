import importlib

def test_progress_surface():
    module = importlib.import_module("cli.progress")
    assert hasattr(module, "get_progress")
    assert hasattr(module, "update")
    assert hasattr(module, "close")
    assert hasattr(module, "set_description")
    assert hasattr(module, "update")
    assert hasattr(module, "tqdm")
    assert hasattr(module, "DummyProgress")
