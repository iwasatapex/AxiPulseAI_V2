import importlib

def test_main_surface():
    module = importlib.import_module("backups.safe_repair_20260808_045901.api.main")
    assert hasattr(module, "http_exception_handler")
    assert hasattr(module, "validation_exception_handler")
    assert hasattr(module, "root")
    assert hasattr(module, "health_check")
