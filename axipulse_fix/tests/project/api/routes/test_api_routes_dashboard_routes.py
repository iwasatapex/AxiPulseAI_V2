import importlib

def test_dashboard_routes_surface():
    module = importlib.import_module("api.routes.dashboard_routes")
    assert hasattr(module, "get_dashboard")
