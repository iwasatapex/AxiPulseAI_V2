import importlib

def test_api_key_surface():
    module = importlib.import_module("api.security.api_key")
    assert hasattr(module, "verify_api_key")
