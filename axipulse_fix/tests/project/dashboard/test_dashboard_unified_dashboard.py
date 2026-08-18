import importlib

def test_unified_dashboard_surface():
    module = importlib.import_module("dashboard.unified_dashboard")
    assert hasattr(module, "check_api")
