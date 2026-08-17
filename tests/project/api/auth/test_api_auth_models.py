import importlib

def test_models_surface():
    module = importlib.import_module("api.auth.models")
    assert hasattr(module, "User")
    assert hasattr(module, "TokenRequest")
    assert hasattr(module, "TokenResponse")
