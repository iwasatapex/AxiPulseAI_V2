import importlib

def test_api_main_surface():
    module = importlib.import_module("backups.route_import_repair_20260808_050629.api_main")
    assert hasattr(module, "http_exception_handler")
    assert hasattr(module, "validation_exception_handler")
    assert hasattr(module, "root")
    assert hasattr(module, "health_check")
