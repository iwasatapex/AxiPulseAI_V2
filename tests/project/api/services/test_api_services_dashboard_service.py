import importlib

def test_dashboard_service_surface():
    module = importlib.import_module("api.services.dashboard_service")
    assert hasattr(module, "get_dashboard")
    assert hasattr(module, "DashboardService")
