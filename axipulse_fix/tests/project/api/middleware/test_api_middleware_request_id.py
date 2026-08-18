import importlib

def test_request_id_surface():
    module = importlib.import_module("api.middleware.request_id")
    assert hasattr(module, "dispatch")
    assert hasattr(module, "RequestIDMiddleware")
