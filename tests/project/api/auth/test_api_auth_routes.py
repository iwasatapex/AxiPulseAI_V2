import importlib

def test_routes_surface():
    module = importlib.import_module("api.auth.routes")
    assert hasattr(module, "login")
