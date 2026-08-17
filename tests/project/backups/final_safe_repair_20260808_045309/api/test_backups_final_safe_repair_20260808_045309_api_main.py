import importlib

def test_main_surface():
    module = importlib.import_module("backups.final_safe_repair_20260808_045309.api.main")
    assert hasattr(module, "http_exception_handler")
    assert hasattr(module, "validation_exception_handler")
    assert hasattr(module, "root")
    assert hasattr(module, "health_check")
