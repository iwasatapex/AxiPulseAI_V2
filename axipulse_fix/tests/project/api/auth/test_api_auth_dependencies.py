import importlib

def test_dependencies_surface():
    module = importlib.import_module("api.auth.dependencies")
    assert hasattr(module, "current_user")
    assert hasattr(module, "require_admin")
