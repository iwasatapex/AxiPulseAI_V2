import importlib

def test_system_routes_surface():
    module = importlib.import_module("api.routes.system_routes")
    assert hasattr(module, "system_status")
