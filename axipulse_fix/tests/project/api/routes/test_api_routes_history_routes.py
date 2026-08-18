import importlib

def test_history_routes_surface():
    module = importlib.import_module("api.routes.history_routes")
    assert hasattr(module, "decisions")
    assert hasattr(module, "predictions")
