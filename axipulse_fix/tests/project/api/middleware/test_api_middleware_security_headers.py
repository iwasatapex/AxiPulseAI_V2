import importlib

def test_security_headers_surface():
    module = importlib.import_module("api.middleware.security_headers")
    assert hasattr(module, "dispatch")
    assert hasattr(module, "SecurityHeadersMiddleware")
