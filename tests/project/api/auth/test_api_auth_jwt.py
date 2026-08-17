import importlib

def test_jwt_surface():
    module = importlib.import_module("api.auth.jwt")
    assert hasattr(module, "create_token")
    assert hasattr(module, "decode_token")
