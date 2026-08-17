import importlib

def test_connection_surface():
    module = importlib.import_module("api.database.connection")
    assert hasattr(module, "get_db")
