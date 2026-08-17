import importlib

def test_rate_limit_surface():
    module = importlib.import_module("api.middleware.rate_limit")
    assert hasattr(module, "check_rate_limit")
