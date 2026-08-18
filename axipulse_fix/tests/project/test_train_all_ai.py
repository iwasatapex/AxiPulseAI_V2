import importlib

def test_train_all_ai_surface():
    module = importlib.import_module("train_all_ai")
    assert hasattr(module, "banner")
    assert hasattr(module, "section")
    assert hasattr(module, "scan_files")
    assert hasattr(module, "describe_dataset")
    assert hasattr(module, "confirm")
    assert hasattr(module, "show_leaderboard")
    assert hasattr(module, "C")
